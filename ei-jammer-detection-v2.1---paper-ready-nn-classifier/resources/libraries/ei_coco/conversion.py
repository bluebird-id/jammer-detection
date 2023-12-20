import numpy as np
from typing import List, Tuple
import tensorflow as tf

# TODO: this entire class currently uses an untyped version of
#       object detection predictions and ground truth.
#       List[Tuple[tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]]
#       full img, bbox detections and one hot labels
#       it will be ported to use BoundingBoxLabelScore soon.


def normalised_xyxy_to_pixelspace_xywh(xyxy: np.ndarray, width: int, height: int):
    """project normalised xyxy to pixel space xywh"""
    x0, y0, x1, y1 = np.clip(xyxy, 0, 1)  # (0, 1)
    x0, y0, x1, y1 = map(
        int, [x0 * width, y0 * height, x1 * width, y1 * height]
    )  # (0, hw)
    w, h = x1 - x0, y1 - y0
    return [x0, y0, w, h]


def derive_width_height(
    y_true_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]]
):
    """derive height, width from ground truth data"""

    # first check for width height, and ensure it's consistent across dataset
    width, height = None, None
    for img, (_detections, _classes) in y_true_bbox_labels:
        if len(img.shape) != 4:
            raise Exception(
                "expected all images in y_true to be"
                f" shaped (B,H,W,C) but one was {img.shape}"
            )
        if width is None:
            width, height = img.shape[1], img.shape[2]
        else:
            img_shape = img.shape[1], img.shape[2]
            if img_shape != (width, height):
                raise Exception(
                    "inconsistent sizing in y_true images;"
                    f" first was {(width, height)}",
                    f" but another was {img_shape}",
                )
    return width, height


def convert_studio_y_true_to_coco_groundtruth(
    y_true_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]],
    width: int,
    height: int,
    num_classes: int,
):
    images = [{"id": str(i)} for i in range(len(y_true_bbox_labels))]

    categories = [{"id": i} for i in range(num_classes)]

    annotations = []
    annotation_idx = 1
    for img_id, (_img, (detections, classes)) in enumerate(y_true_bbox_labels):
        num_detections = detections.shape[0]
        if classes.shape[0] != num_detections:
            raise Exception(
                f"y_true img_id {img_id} has mismatched #detections vs #labels"
            )
        for detection_id in range(num_detections):
            x, y, w, h = normalised_xyxy_to_pixelspace_xywh(
                detections[detection_id], width, height
            )
            label = np.argmax(classes[detection_id], axis=-1)
            annotations.append(
                {
                    "image_id": str(img_id),
                    "iscrowd": 0,
                    "category_id": label,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "id": annotation_idx,
                }
            )
            annotation_idx += 1

    return {"images": images, "categories": categories, "annotations": annotations}


def convert_studio_y_pred_to_coco_detections(
    y_pred_bbox_labels: List[Tuple[tf.Tensor, tf.RaggedTensor, tf.RaggedTensor]],
    width: int,
    height: int,
):
    coco_predictions = []
    detection_idx = 1

    for img_id, img_detections in enumerate(y_pred_bbox_labels):
        for bbox, label, confidence in img_detections:
            xywh = normalised_xyxy_to_pixelspace_xywh(bbox, width, height)
            coco_predictions.append(
                {
                    "image_id": str(img_id),
                    "category_id": int(label),
                    "bbox": xywh,
                    "score": confidence,
                    "id": detection_idx,
                }
            )
            detection_idx += 1

    return coco_predictions
