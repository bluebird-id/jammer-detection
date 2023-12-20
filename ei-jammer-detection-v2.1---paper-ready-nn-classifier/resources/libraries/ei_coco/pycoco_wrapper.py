# ported from keras-cv
# keras_cv/metrics/coco/pycoco_wrapper.py r0.5.0

# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from pycocotools.coco import COCO

class PyCOCOWrapper(COCO):
    """COCO wrapper class.
    This class wraps COCO API object, which provides the following additional
    functionalities:
      1. Support string type image id.
      2. Support loading the groundtruth dataset using the external annotation
         dictionary.
      3. Support loading the prediction results using the external annotation
         dictionary.
    """

    def __init__(self, gt_dataset=None):
        """Instantiates a COCO-style API object.
        Args:
          eval_type: either 'box' or 'mask'.
          annotation_file: a JSON file that stores annotations of the eval
            dataset. This is required if `gt_dataset` is not provided.
          gt_dataset: the groundtruth eval datatset in COCO API format.
        """
        COCO.__init__(self, annotation_file=None)
        self._eval_type = "box"
        if gt_dataset:
            self.dataset = gt_dataset
            self.createIndex()

    def loadRes(self, predictions):
        """Loads result file and return a result api object.
        Args:
          predictions: a list of dictionary each representing an annotation in
            COCO format. The required fields are `image_id`, `category_id`,
            `score`, `bbox`, `segmentation`.
        Returns:
          res: result COCO api object.
        Raises:
          ValueError: if the set of image id from predictions is not the subset
            of the set of image id of the groundtruth dataset.
        """
        res = COCO()
        res.dataset["images"] = copy.deepcopy(self.dataset["images"])
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])

        image_ids = [ann["image_id"] for ann in predictions]
        if set(image_ids) != (set(image_ids) & set(self.getImgIds())):
            raise ValueError(
                "Results do not correspond to the current dataset!"
            )
        for ann in predictions:
            x1, x2, y1, y2 = [
                ann["bbox"][0],
                ann["bbox"][0] + ann["bbox"][2],
                ann["bbox"][1],
                ann["bbox"][1] + ann["bbox"][3],
            ]

            ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

        res.dataset["annotations"] = copy.deepcopy(predictions)
        res.createIndex()
        return res