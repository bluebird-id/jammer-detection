import tensorflow as tf
import numpy as np
from sklearn import metrics as sklearn_metrics
import warnings
from typing import List, Tuple

import ei_tensorflow.constrained_object_detection.metrics as fomo_metrics


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate a collection of regression specific metrics.

    Args:
        y_true: ground truth values
        y_pred: predictions
    Returns:
        a dict containing a collection of sklearn regression metrics.
    """

    # TODO: are there any conditions where the following throw exceptions?
    #       i.e. should we run each in try/except and build dict up?
    return {
        "mean_squared_error": sklearn_metrics.mean_squared_error(y_true, y_pred),
        "mean_absolute_error": sklearn_metrics.mean_absolute_error(y_true, y_pred),
        "explained_variance_score": sklearn_metrics.explained_variance_score(
            y_true, y_pred
        ),
    }


def calculate_classification_metrics(
    y_true_one_hot: np.ndarray, y_pred_probs: np.ndarray, num_classes: int
):
    """Calculate a collection of classification specific metrics.

    Args:
        y_true_one_hot: ground truth values in one hot format.
        y_pred_probs: predictions containing full probability distribution
        num_classes: total number of classes
    Returns:
        a dict containing a collection of sklearn classification metrics.
    """

    # TODO: derive num_classes from width of y_true_one_hot?

    # sanity check y_true_one_hot look one hot
    # or at least close to one hot, since for int8 model values
    # they end up being a tad off 1.0 :/
    row_wise_sums = np.sum(y_true_one_hot, axis=-1)
    difference_from_1 = np.abs(1 - row_wise_sums)
    if difference_from_1.max() > 1e-2:
        print(
            f"WARNING: y_true_one_hot provided does not look one hot {difference_from_1.max()}"
        )

    # always renormalise y_pred_probs. we do this int8 values can
    # give a result that fails the atol tests of roc_auc
    y_pred_probs /= y_pred_probs.sum(axis=-1, keepdims=True)

    # build a labels list, [0, 1, 2, ...] which is used by
    # a number of the sklearn metrics
    labels = list(range(num_classes))

    # convert from distribution to labels for some metrics
    y_true_labels = y_true_one_hot.argmax(axis=-1)
    y_pred_labels = y_pred_probs.argmax(axis=-1)

    metrics = {}

    metrics["confusion_matrix"] = sklearn_metrics.confusion_matrix(
        y_true_labels, y_pred_labels, labels=labels
    ).tolist()

    metrics["classification_report"] = sklearn_metrics.classification_report(
        y_true_labels, y_pred_labels, output_dict=True, zero_division=0
    )

    try:
        if num_classes == 2:
            # NOTE! roc_auc calculation for binary case must be called with
            #       labels otherwise it throws an exception
            metrics["roc_auc"] = sklearn_metrics.roc_auc_score(
                y_true=y_true_labels, y_score=y_pred_labels, multi_class="ovr"
            )
        else:
            metrics["roc_auc"] = sklearn_metrics.roc_auc_score(
                y_true=y_true_labels, y_score=y_pred_probs, multi_class="ovr"
            )
    except Exception as e:
        # a known common case for this is when not all classes are
        # represented. we can detect this from the exception and ignore. but
        # if it's something else we should reraise
        if (
            "Number of classes in y_true not"
            " equal to the number of columns in 'y_score'" in str(e)
        ):
            metrics["roc_auc"] = None
        else:
            raise e

    metrics["loss"] = sklearn_metrics.log_loss(
        y_true_labels, y_pred_probs, labels=labels
    )

    return metrics


def _coco_map_calculation_from_studio(
    y_true_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor]],
    prediction: List[Tuple[tf.Tensor, tf.RaggedTensor]],
    num_classes: int
):
    # coco map calculation taken as is from ei_tensorflow.profiling.
    # keep as seperate method to denote code copied as is

    # This is only installed on object detection containers so import it only when used
    from mean_average_precision import MetricBuilder

    metric_fn = MetricBuilder.build_evaluation_metric(
        "map_2d", async_mode=True, num_classes=num_classes
    )

    # Calculate mean average precision
    def un_onehot(onehot_array):
        """Go from our one-hot encoding to an index"""
        val = np.argmax(onehot_array, axis=0)
        return val

    for index, sample in enumerate(y_true_bbox_labels):
        labels = sample[1]
        p = prediction[index]
        gt = []
        curr_ps = []

        boxes = labels[0]
        labels = labels[1]
        for box_index, box in enumerate(boxes):
            label = labels[box_index]
            label = un_onehot(label)
            # The library expects [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            # We use the TF standard [ymin, xmin, ymax, xmax], so translate it
            gt.append([box[1], box[0], box[3], box[2], label, 0, 0])

        for p2 in p:
            # The library expects [xmin, ymin, xmax, ymax, class_id, confidence]
            curr_ps.append([p2[0][1], p2[0][0], p2[0][3], p2[0][2], p2[1], p2[2]])

        gt = np.array(gt)
        curr_ps = np.array(curr_ps)

        metric_fn.add(curr_ps, gt)

    coco_map = metric_fn.value(
        iou_thresholds=np.arange(0.5, 1.0, 0.05),
        recall_thresholds=np.arange(0.0, 1.01, 0.01),
        mpolicy="soft",
    )["mAP"]

    return float(coco_map)


def calculate_object_detection_metrics(
    y_true_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor]],
    y_pred_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor]],
    num_classes: int
):
    """Calculate a collection of object detection specific metrics.

    Args:
        y_true_bbox_labels: ground truth values contained bounding boxes and labels
        y_pred: bounding box predictions.
        num_classes: total number of classes
    Returns:
        a dict containing a collection of sklearn object detection metrics.
    """

    # TODO: review with the boundingBoxLabel class introduced with FOMO work,
    #       we might be able to get rid of some more code.

    metrics = {}

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        metrics["coco_map"] = _coco_map_calculation_from_studio(
            num_classes=num_classes,
            y_true_bbox_labels=y_true_bbox_labels,
            prediction=y_pred_bbox_labels,
        )

    # TODO: add IOU ? others?

    return metrics


def calculate_fomo_metrics(
    y_true_labels: List[int], y_pred_labels: List[int], num_classes: int
):
    """Calculate a collection of FOMO specific metrics.
    These are calculated seperately from object detection because, fundamentally,
    FOMO is more like NxN classification and segmentation than bounding box
    detection.

    Args:
        y_true_labels: ground truth labels flattened from (H,W,C) to single list of length H*W*C.
        y_pred_labels: predicted labels flattened from (H,W,C) to single list of length H*W*C.
        num_classes: total number of classes.
    Returns:
        a dict containing a collection of sklearn & fomo specific detection metrics.
    """

    metrics = {}

    metrics["confusion_matrix"] = sklearn_metrics.confusion_matrix(
        y_true_labels, y_pred_labels, labels=range(num_classes)
    ).tolist()

    precision, recall, f1 = fomo_metrics.non_background_metrics(
        y_true_labels, y_pred_labels, num_classes
    )
    metrics["non_background"] = {"precision": precision, "recall": recall, "f1": f1}

    metrics["classification_report"] = sklearn_metrics.classification_report(
        y_true_labels, y_pred_labels, output_dict=True, zero_division=0
    )

    return metrics
