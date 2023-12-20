from __future__ import annotations
from typing import Literal, NamedTuple, Optional, Any
from types import SimpleNamespace
import json

class TrainInput(NamedTuple):
    classes: list[str]
    mode: Literal['classification', 'regression', 'object-detection']
    printHWInfo: Optional[bool]
    inputShape: tuple[int]
    inputShapeString: str
    yType: Literal['npy', 'structured']
    trainTestSplit: float
    stratifiedTrainTest: bool
    onlineDspConfig: Optional[Any]
    convertInt8: bool
    objectDetectionLastLayer: Optional[Literal['mobilenet-ssd', 'fomo', 'yolov5', 'yolov5v5-drpai', 'yolox', 'yolov7']]
    objectDetectionAugmentation: Optional[bool]
    # Batch size is provided here when training SSD object detection models,
    # but not used for other models.
    objectDetectionBatchSize: Optional[int]
    syntiantTarget: Optional[bool]
    maxTrainingTimeSeconds: int
    remainingGpuComputeTimeSeconds: int
    isEnterpriseProject: bool

def parse_train_input(file: str) -> TrainInput:
    with open(file, 'r') as f:
        # the object_hook here makes proper keys of every property, so rather than:
        # x['mode'], you can type x.mode, like a normal well-behaved person
        x = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
        return x

def parse_input_shape(s: str) -> tuple[int, ...]:
    # the inputShapeString comes in as the string "(33,3,)" - so turn it into a proper tuple (33,3)
    input_shape = tuple([ int(x) for x in list(filter(None, s.replace('(', '').replace(')', '').split(','))) ])
    return input_shape
