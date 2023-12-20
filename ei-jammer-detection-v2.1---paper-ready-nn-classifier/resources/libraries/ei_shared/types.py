import json, os
from typing import Literal, Optional

# This should be kept identical to ClassificationMode in keras-types.ts.
ClassificationMode = Literal[
    "classification", "regression", "object-detection", "visual-anomaly", "anomaly-gmm"
]

# These should be kept identical to the equivalents in deployment-types.ts.
TAOObjectDetectionLastLayers = Literal[
    "tao-retinanet", "tao-ssd", "tao-yolov3", "tao-yolov4"
]
ObjectDetectionLastLayer = Literal[
    "mobilenet-ssd",
    "fomo",
    "yolov2-akida",
    "yolov5",
    "yolov5v5-drpai",
    "yolox",
    "yolov7",
    TAOObjectDetectionLastLayers,
]


class ObjectDetectionDetails:
    """
    Contains the details required to run an object detection model.
    """

    def __init__(
        self,
        last_layer: ObjectDetectionLastLayer,
        tao_nms_attributes: Optional[dict] = None,
    ):
        self.last_layer = last_layer
        self.tao_nms_attributes = tao_nms_attributes

    @classmethod
    def create(
        cls,
        mode: ClassificationMode,
        last_layer: Optional[ObjectDetectionLastLayer],
        tao_nms_attributes: Optional[dict] = None,
        tao_nms_path: Optional[str] = None,
    ):
        """
        Loads the object detection details from the given input.
        Pass either a tao_nms_attributes dictionary, or a tao_nms_path to load the attributes from a file.
        If the mode is not object-detection then None is returned.
        """
        if mode != "object-detection":
            return None

        if last_layer is None:
            raise ValueError("last_layer cannot be none if mode is object-detection")

        if tao_nms_attributes is not None and tao_nms_path is not None:
            raise ValueError(
                "Specify either tao_nms_attributes or tao_nms_path, but not both."
            )

        # If the file exists then this is a TAO object detection model
        if tao_nms_path is not None and os.path.exists(tao_nms_path):
            try:
                with open(tao_nms_path, "r") as f:
                    tao_nms_attributes = json.load(f)
            except Exception as err:
                print(
                    f"WARN: Failed to load TAO NMS attributes from {tao_nms_path}", err
                )
                raise err

        return ObjectDetectionDetails(
            last_layer=last_layer, tao_nms_attributes=tao_nms_attributes
        )

    def tao_nms(self, key: str):
        if not self.last_layer.startswith("tao"):
            raise RuntimeError(
                f"last_layer is '{self.last_layer}', but tao_nms() is only supported for tao last layers"
            )
        if self.tao_nms_attributes is None:
            raise RuntimeError("There are no TAO NMS attributes specified")
        return self.tao_nms_attributes[key]
