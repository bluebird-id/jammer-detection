import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from typing import List
from collections import namedtuple
from scipy.special import softmax
import time

from ei_tensorflow.constrained_object_detection.util import batch_convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.util import convert_from_ragged
from ei_shared.labels import BoundingBoxLabelScore

def non_background_metrics_from_confusion(confusion_matrix: np.array,
                                          background_class: int=0):
    """ Calculate binary precision, recall & f1 for background vs not background classes.

    Args:
        confusion_matrix: as calculated by sklearn confusion_matrix
        background_class: which index in the confusion matrix represents the
          background. all other classes combined into one.

    Returns:
        tuple of precision, recall & f1
    """
    if len(confusion_matrix.shape) != 2:
        raise Exception(f"Invalid confusion matrix; expected 2d, not {confusion_matrix}")
    n_rows, n_cols = confusion_matrix.shape
    if n_rows != n_cols or n_rows < 2:
        raise Exception(f"Invalid confusion matrix; expected at least two classes. {confusion_matrix.shape}")

    # true negatives; i.e. background, is just the count of the background_class entry
    true_negatives = confusion_matrix[background_class, background_class]
    # true positives is the sum of the main diagonal entries, minus the background
    true_positives = np.diagonal(confusion_matrix).sum() - true_negatives
    # number of false negatives is the sum of the lower triangle, exluding main diagonal
    false_negatives = np.tril(confusion_matrix, k=-1).sum()
    # number of false positives is the sum of the upper triangle, exluding main diagonal
    false_positives = np.triu(confusion_matrix, k=1).sum()

    # in the case of only true negatives return 1.0 for P/R/F1
    if true_positives == 0 and false_negatives == 0 and false_positives == 0:
        return 1.0, 1.0, 1.0

    # calculate precision, recall and f1. return 0.0 for ill defined cases.
    precision = 0.0 if (true_positives + false_positives == 0) else true_positives / (true_positives + false_positives)
    recall = 0.0 if (true_positives + false_negatives == 0) else true_positives / (true_positives + false_negatives)
    f1 = 0.0 if (precision + recall == 0) else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def non_background_metrics(y_true: np.array, y_pred: np.array,
                           num_classes: int):
    confusion = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    return non_background_metrics_from_confusion(confusion)

class PrintPercentageTrained(Callback):
    """ Callback to regularly print % trained to stdout. """

    def __init__(self, num_epochs: int):
        self.num_epochs = num_epochs
        self.first_epoch = True  # record number of batches during first epoch.
        self.total_num_batches_trained = 0
        self.num_batches_per_epoch = 0
        self.last_update_time = time.time()

    def on_batch_end(self, epoch, logs):
        self.total_num_batches_trained += 1
        if self.first_epoch:
            self.num_batches_per_epoch += 1
        if time.time() > self.last_update_time + 20:
            if self.first_epoch:
                print(f"Trained {self.total_num_batches_trained} batches.")
            else:
                percentage = float(self.total_num_batches_trained) / self.total_batches * 100
                print(f"{percentage:.1f}% trained.")
            self.last_update_time = time.time()

    def on_epoch_end(self, epoch, logs):
        # calculate expected total number of batches at end of first epoch.
        if self.first_epoch:
            self.total_batches = self.num_epochs * self.num_batches_per_epoch
            self.first_epoch = False
        # since epoch end has an implicit print update we reset this
        # update time to avoid spamming stdout.
        self.last_update_time = time.time()

class CentroidScoring(Callback):
    """ A callback for centroid scoring on validation data on epoch end. """

    def __init__(self,
                 validation_dataset: tf.data.Dataset,  # ragged
                 output_width_height: int,
                 num_classes_including_background: int):

        self.dataset = validation_dataset

        self.output_width_height = output_width_height
        self.num_classes_including_background = num_classes_including_background

    def on_epoch_end(self, epoch, logs):
        # run model over validation data. recall; model is just logits so need
        # to run softmax before conversion.
        y_pred = self.model.predict(self.dataset, verbose=0)
        y_pred = softmax(y_pred, axis=-1)

        # convert to bounding box label scores for near centroid matching.
        y_pred = batch_convert_segmentation_map_to_object_detection_prediction(
                                y_pred, minimum_confidence_rating=0.5, fuse=True)

        # do alignment by centroids. this results in a flatten list of int
        # labels that is suitable for confusion matrix calculations.
        y_true_labels, y_pred_labels = dataset_match_by_near_centroids(
            self.dataset, y_pred, self.output_width_height)

        val_precision, val_recall, val_f1 = non_background_metrics(
            y_true_labels, y_pred_labels,
            self.num_classes_including_background)

        logs['val_precision'] = val_precision
        logs['val_recall'] = val_recall
        logs['val_f1'] = val_f1

        print()
        print("Epoch   Train    Validation")
        print("        Loss     Loss    Precision Recall F1")
        print(f"   {epoch:02d}" +
                f"   {logs['loss']:.05f}" +
                f"  {logs['val_loss']:.05f}" +
                f" {val_precision:.02f}" +
                f"      {val_recall:.02f}"
                f"   {val_f1:.02f}")

Assignment = namedtuple('Assignment', ['yp', 'yt', 'label', 'distance'])

def match_by_near_centroids(
    y_trues: List[BoundingBoxLabelScore],
    y_preds: List[BoundingBoxLabelScore],
    min_normalised_distance: float,
    output_width_height: int,
    return_debug_info: bool=False) -> (List[int], List[int]):
    """ Match a set of y_pred and y_true based on nearby centroids.

    Args:
        y_trues: A list of BoundingBoxLabelScores. coordinates are normalised
            and labels include implicit class=0.
        y_preds: A list of BoundingBoxLabelScores. coordinates are normalised
            and labels include implicit class=0.
        min_normalised_distance: minimum distance for a match expressed as
            normalised (0.0, 1.0) value.
        output_width_height: size of output, required to derive the implied
            true negative count.
        return_debug_info: whether to return a dictionary of matching
            debug information.

    Returns:
        a tuple of y_true and y_pred lists. these lists are a flatten list
        of labels suitable for use in confusion matrix calculation.
    """

    # We do matching in label space that includes implicit class=0 background
    # and never expect to get background labelled data. Do this check to avoid
    # potential off-by-1 errors in label manipulation.
    for eg in y_trues:
        if eg.label == 0:
            raise Exception(f"Didn't expect to have labelled background from {y_trues}")
    for eg in y_preds:
        if eg.label == 0:
            raise Exception(f"Didn't expect to have labelled background from {y_preds}")

    # check min_normalised_distance is normalised value
    if min_normalised_distance < 0.0 or min_normalised_distance > 1.0:
        raise Exception("min_normalised_distance must be in range (0, 1)"
                        f" not {min_normalised_distance}")

    # nothing in either y_true or y_pred results in all true negatives
    num_cells = output_width_height * output_width_height
    if len(y_trues) == 0 and len(y_preds) == 0:
        y_true_labels = [0] * num_cells
        y_pred_labels = y_true_labels
        if return_debug_info:
            no_debug_info = {'assignments': [],
                             'unassigned_y_true_idxs': [],
                             'unassigned_y_pred_idxs': []}
            return y_true_labels, y_pred_labels, no_debug_info
        else:
            return y_true_labels, y_pred_labels

    # convert both y_true and y_pred to centroids
    y_trues = [bls.centroid() for bls in y_trues]
    y_preds = [bls.centroid() for bls in y_preds]

    # keep track of which items haven't been assigned, the counts of these
    # sets will be the false negative/positive counts
    unassigned_y_true_idxs = set(range(len(y_trues)))
    unassigned_y_pred_idxs = set(range(len(y_preds)))

    # compare each y_pred to all y_true. any y_pred close enough to a
    # y_true will be deemed correct; even if this means N y_pred to 1 y_true
    all_pairwise_distances = []
    assignments = []
    for yp, y_pred in enumerate(y_preds):
        best_assignment = None
        best_assignment_distance = 1e10
        for yt, y_true in enumerate(y_trues):
            if y_pred.label == y_true.label:
                distance = y_pred.distance_to(y_true)
                all_pairwise_distances.append((yp, yt, distance))
                if distance <= min_normalised_distance and distance < best_assignment_distance:
                    best_assignment = Assignment(yp, yt, y_pred.label, distance)
                    best_assignment_distance = distance
        if best_assignment is not None:
            assignments.append(best_assignment)
            unassigned_y_pred_idxs.discard(best_assignment.yp)
            unassigned_y_true_idxs.discard(best_assignment.yt)

    # synthetically construct a flat list of y_true and y_pred labels for return.
    # these lists will includes 0 as a background class to allow false positives
    # calculation.
    y_true_labels = []
    y_pred_labels = []

    # each assignment is considered a true positive case
    for assignment in assignments:
        y_true_labels.append(assignment.label)
        y_pred_labels.append(assignment.label)

    # the y_pred values that weren't close enough to a matching y_true
    # are considered false positives
    for yp in unassigned_y_pred_idxs:
        y_true_labels.append(0)
        y_pred_labels.append(y_preds[yp].label)

    # the y_true values that weren't matched to a y_pred value are considered
    # false negatives
    for yt in unassigned_y_true_idxs:
        y_true_labels.append(y_trues[yt].label)
        y_pred_labels.append(0)

    # the number of remaining cells in the output are considered true negatives
    while len(y_true_labels) < num_cells:
        y_true_labels.append(0)
        y_pred_labels.append(0)

    # main return value is the lists of y_true and y_pred int labels
    if not return_debug_info:
        return y_true_labels, y_pred_labels

    # also return matching info for debug visualisation
    debug_info = {
        'y_trues': y_trues, 'y_preds': y_preds,
        'normalised_min_distance': min_normalised_distance,
        'assignments': assignments,
        'all_pairwise_distances': all_pairwise_distances,
        'unassigned_y_true_idxs': list(unassigned_y_true_idxs),
        'unassigned_y_pred_idxs': list(unassigned_y_pred_idxs)
    }
    return y_true_labels, y_pred_labels, debug_info

def dataset_match_by_near_centroids(dataset, y_preds, output_width_height):
    y_true_labels = []
    y_pred_labels = []
    i = 0
    for items in dataset:
        # During validation the dataset contains three variables,
        # but in profiling there are only two.
        length = len(items)
        if length == 3:
            x, y_map, (boxes, classes) = items
        elif length == 2:
            x, (boxes, classes) = items
        else:
            raise Exception('Expected at least two variables in dataset item')
        batch_bbox_label_score = convert_from_ragged(boxes, classes,
                                                     offset_label_by_one=True)
        for bbls in batch_bbox_label_score:
            y_p = y_preds[i]
            instance_y_true_labels, instance_y_pred_labels = match_by_near_centroids(
                bbls, y_p, output_width_height=output_width_height,
                min_normalised_distance=0.2)
            y_true_labels.extend(instance_y_true_labels)
            y_pred_labels.extend(instance_y_pred_labels)
            i += 1

    return y_true_labels, y_pred_labels

def debug_image(x: np.ndarray,
                y_trues: List[BoundingBoxLabelScore],
                y_preds: List[BoundingBoxLabelScore],
                debug_info: dict):        # from match_by_near_centroids

    from PIL import Image, ImageDraw
    DEBUG_HW = 256
    x = (x*255).astype(np.uint8)
    img = Image.fromarray(x, 'L').convert('RGB').resize((DEBUG_HW, DEBUG_HW))
    draw = ImageDraw.Draw(img, "RGBA")

    # to draw on pil iamge we need to
    # 1) project() from normalised coordinates to DEBUG image size
    # 2) floor() the values to ints
    # #) transpose_x_y() since PIL coords are transposed

    def draw_box(bbox, colour):
        bbox = bbox.project(DEBUG_HW, DEBUG_HW).floored().transpose_x_y()
        cx, cy = tuple(bbox.centroid())
        draw.rectangle(list(bbox), outline=colour, width=5)
        draw.rectangle((cx-2,cy-2,cx+2,cy+2), outline=colour, width=2)

    # true positives are green
    for a in debug_info['assignments']:
        draw_box(y_preds[a.yp].bbox, 'blue')
        draw_box(y_trues[a.yt].bbox, 'green')
    # false positives are red
    for yp in debug_info['unassigned_y_pred_idxs']:
        draw_box(y_preds[yp].bbox, 'red')
    # false negatives are purple
    for yt in debug_info['unassigned_y_true_idxs']:
        draw_box(y_trues[yt].bbox, 'purple')
    return img
