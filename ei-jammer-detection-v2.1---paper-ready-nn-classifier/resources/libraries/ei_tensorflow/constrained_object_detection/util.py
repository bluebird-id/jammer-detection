import tensorflow as tf
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import numpy as np
import math
from ei_shared.labels import BoundingBoxLabelScore, BoundingBox
from typing import List

def logit(x):
    return np.log(x/(1-x))

def bbox_range_check(x0: float, y0: float, x1: float, y1: float):
    """ Check range values of bounding box values."""
    if x0 < 0.0 or x0 > 1.0: raise Exception("x0", x0, "not in range (0, 1)")
    if y0 < 0.0 or y0 > 1.0: raise Exception("y0", y0, "not in range (0, 1)")
    if x1 < 0.0 or x1 > 1.0: raise Exception("x1", x1, "not in range (0, 1)")
    if y1 < 0.0 or y1 > 1.0: raise Exception("y1", y1, "not in range (0, 1)")
    if x0 > x1: raise Exception("expected x0", x0, "to be < x1", x1)
    if y0 > y1: raise Exception("expected y0", y0, "to be < y1", y1)

def convert_bounding_boxes_to_mask(bboxes: List[BoundingBox],
                                   height_width: int) -> np.ndarray:
    mask = np.zeros((height_width, height_width), dtype=np.uint8)
    for bbox in bboxes:
        bbox = bbox.floored()
        mask[bbox.x0:bbox.x1, bbox.y0:bbox.y1] = 1
    return mask

def convert_from_ragged(bboxes_batch: RaggedTensor,
                        labels_batch: RaggedTensor,
                        offset_label_by_one: bool=False) -> List[List[BoundingBoxLabelScore]]:
    """ Converts from TF object detection ragged format to a collection of BoundingBoxLabelScores.
    Args:
        bboxes_batch: ragged tensor of shape (B, None, 4) representing the 0+ bounding boxes
            for a batch of B images. box values are assumed to be (x0, y0, x1, x2)
            and normalised, i.e. values (0.0, 1.0)
        labels_batch: ragged tensor of shape (B, None, num_classes) representing the one
            hot distribution of labelled classes. this num_classes size is used
            to decide the size of the output.
        offset_label_by_one: whether to add 1 to label to make it ready for
            processing with respect to an implicit background class.
    Returns:
        a list of lists of BoundingBoxLabelScore. the outer list representing the
        batch, the inner list representing the detections per element of the batch
    """
    batch_bbox_label_scores = []

    # explicitly convert the batch to numpy() _before_ any iteration
    # otherwise it seems this corrupts the ragged tensor on GPU
    bboxes_batch = bboxes_batch.numpy()
    labels_batch = labels_batch.numpy()

    for bboxes, one_hot_labels in zip(bboxes_batch, labels_batch):
        # un one hotify the labels
        _idxs, labels = np.where(one_hot_labels==1.0)

        # convert to list of BoundingBoxLabelScores
        bbox_label_scores = []
        for bbox, label in zip(bboxes, labels):
            x0, y0, x1, y1 = map(float, bbox)
            label = int(label)
            if offset_label_by_one:
                label += 1
            bbox_label_scores.append(
                BoundingBoxLabelScore(
                    BoundingBox(x0, y0, x1, y1),
                    label=label))

        # add to batch
        batch_bbox_label_scores.append(bbox_label_scores)

    return batch_bbox_label_scores

def convert_sample_bbox_and_labels_to_boundingboxlabelscores(
    bboxes_dict: list,
    input_width_height: int) -> List[BoundingBoxLabelScore]:
    bls = []
    for d in bboxes_dict:
        # ignore entry if height or width is zero
        if d['h'] == 0 or d['w'] == 0:
            continue
        # map from x,y,w,h to x0,y0,x1,y1
        bbox = BoundingBox.from_x_y_h_w(d['x'], d['y'], d['h'], d['w'])
        # sample from UI has x/y flipped with respect to ragged data so flip now
        bbox = bbox.transpose_x_y()
        # normalise
        bbox = bbox.project(1/input_width_height, 1/input_width_height)
        # note: this data already covers a background class of 0 so we should
        # enver expect to be sent a background labelled instance.
        if d['label'] == 0:
            raise Exception("Never expect sample from studio to have label==0")
        # collect with 0 based index label and score 1.0
        bls.append(BoundingBoxLabelScore(bbox, label=d['label'], score=1.0))
    return bls

def fuse_adjacent(bbox_label_scores: List[BoundingBoxLabelScore]) -> List[BoundingBoxLabelScore]:
    """ Fuse adjacent / overlapping bboxes in the same way ei_cube_check_overlap does. """
    if bbox_label_scores == []:
        return []
    collected_bbox_label_scores = []
    for orig_bls in bbox_label_scores:
        had_overlap = False
        for collected_bls in collected_bbox_label_scores:
            if collected_bls.label != orig_bls.label:
                continue
            had_overlap = collected_bls.bbox.update_with_overlap(orig_bls.bbox)
            if had_overlap:
                if orig_bls.score > collected_bls.score:
                    collected_bls.score = orig_bls.score
                break
        if not had_overlap:
            collected_bbox_label_scores.append(orig_bls)
    return collected_bbox_label_scores

# TODO(mat): rename away from to_object_detection_prediction
def convert_segmentation_map_to_object_detection_prediction(
    segmentation_map: np.ndarray,
    minimum_confidence_rating: float,
    fuse: bool) -> List[BoundingBoxLabelScore]:
    """ Converts (N, N) segmentation map back to bbox and one hot labels.

    Args:
      segmentation_map: (H, W, C) segmentation map output.
      minimum_confidence_rating: threshold for probability.
      fuse: whether to fuse adjacent cells in a manner matching
        ei_cube_check_overlap

    Returns:
      list of (bbox, label, score) tuples representing detections.

    Given a (H, W, C) output from a segmentation model convert back
    to list of (bbox, label, score) as used in Studio. Filter entries to
    have at least minimum_confidence_rating probability.
    """

    # check shape
    if len(segmentation_map.shape) != 3:
       raise Exception("Expected segmentation map to be shaped "
                        f" (H, W, C) but was {segmentation_map.shape}")
    width, height, num_classes_including_background = segmentation_map.shape

    # check it has some non background classes
    if num_classes_including_background < 2:
        raise Exception("Expected at least one non background class but"
                        f" had {num_classes_including_background}"
                        f" (shape {segmentation_map.shape})")

    # will return boxes, labels and scores
    boxes_labels_scores = []

    # check all non background classes (background is class 0)
    for class_idx in range(1, num_classes_including_background):
        # TODO(mat): should we fuse and THEN filter by min conf rating?
        # determine which grid points are at least the minimum confidence rating
        xs, ys = np.where(segmentation_map[:,:,class_idx] > minimum_confidence_rating)
        for x, y in zip(xs, ys):
            # retain class 0 as background
            boxes_labels_scores.append(
                BoundingBoxLabelScore(
                    BoundingBox(x/width, y/height, (x+1)/width, (y+1)/height),
                    label=class_idx,
                    score=float(segmentation_map[x, y, class_idx])),
            )

    if fuse:
        boxes_labels_scores = fuse_adjacent(boxes_labels_scores)

    return boxes_labels_scores

def batch_convert_segmentation_map_to_object_detection_prediction(
    segmentation_maps: np.ndarray,
    minimum_confidence_rating: float,
    fuse: bool) -> List:

    # check shape
    if len(segmentation_maps.shape) != 4:
       raise Exception("expected segmentation map to be shaped "
                        f" (B, H, W, C) but was {segmentation_map.shape}")

    # process batch by running each entry through
    # convert_segmentation_map_to_object_detection_prediction
    batch_boxes_labels_scores = []
    for segmentation_map in segmentation_maps:
        # collect boxes_labels_scores into batch list
        batch_boxes_labels_scores.append(
            convert_segmentation_map_to_object_detection_prediction(
                segmentation_map, minimum_confidence_rating, fuse))
    return batch_boxes_labels_scores




def set_classifier_biases_from_dataset(model,
                                       dataset: tf.data.Dataset,
                                       # A previous version of this function required a third arg.
                                       # Allow it for compatibility with legacy code.
                                       *args):
    """ Set the final layer bias values based on dataset label distribution. """
    try:
        # extract weights from last layer
        classifier_layer = model.layers[-1]
        weights, _biases = classifier_layer.get_weights()

        # use sizing of last layer to infer num_classes
        num_classes_with_background = weights.shape[-1]

        # collect label frequency from the first 10 batches of the dataset.
        # use laplace smoothing (i.e. default 1 count) to avoid pathological no
        # labelled objects case.
        class_counts = np.ones(num_classes_with_background)
        for i, (_x, y) in enumerate(dataset):
            # y is one hot so class distribution is argmax of outer axis
            one_hot_labels = np.array(y).reshape((-1, num_classes_with_background))
            for label in one_hot_labels.argmax(axis=-1):
                class_counts[label] += 1
            if i > 10:
                break

        # calculate the corresponding logits of the proportions
        proportions = class_counts / np.sum(class_counts)
        logits = logit(proportions)

        # set these logits as the bias values of the last layer in model
        classifier_layer.set_weights((weights, logits))

    except Exception as e:
        print(f"Warning: Unable to set initial biases [{e}]")