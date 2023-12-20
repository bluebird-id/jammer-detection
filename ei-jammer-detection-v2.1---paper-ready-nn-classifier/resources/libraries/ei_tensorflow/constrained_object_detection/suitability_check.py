import json, math
import numpy as np
from PIL import Image, ImageDraw
from ei_shared.labels import BoundingBox, Centroid
from ei_tensorflow.constrained_object_detection import util
from typing import List, Tuple, Dict
from tensorflow.python.data.ops.dataset_ops import Dataset, BatchDataset

class FOMOSuitabilityCheck(object):

    @staticmethod
    def from_studio_files(x_npy: str,
                          y_json: str,
                          fomo_reduction: int):
        # create FOMOSuitabilityCheck given paths to data and label files
        # as downloaded from studio dashboard.

        x = np.load(x_npy)
        x = x.reshape(FOMOSuitabilityCheck.infer_square_shape(x))
        x = np.squeeze(x)  # pil wants to load grayscale from (h,w) not (h,w,1)
        HW = x.shape[1]

        y = json.loads(open(y_json).read())
        bboxes = []
        for sample in y['samples']:
            boundbox_label_scores = util.convert_sample_bbox_and_labels_to_boundingboxlabelscores(
                bboxes_dict=sample['boundingBoxes'],
                input_width_height=HW)
            bboxes.append([e.bbox for e in boundbox_label_scores])

        return FOMOSuitabilityCheck(x, bboxes, fomo_reduction)

    @staticmethod
    def from_expert_mode_tf_dataset(dataset: Dataset,
                                    fomo_reduction: int):
        # create FOMOSuitabilityCheck given the datasets available in expert mode.

        if type(dataset) != BatchDataset:
            # convert_from_ragged expected batched data
            dataset = dataset.batch(1)

        x, bboxes = [], []
        for ds_x, (ds_bboxes, ds_labels) in dataset:
            x.append(ds_x.numpy())
            boundbox_label_scores = util.convert_from_ragged(ds_bboxes, ds_labels)
            assert len(boundbox_label_scores) == 1
            boundbox_label_scores = boundbox_label_scores[0]  # strip single batch
            just_bboxes = [e.bbox for e in boundbox_label_scores]
            bboxes.append(just_bboxes)
        x = np.concatenate(x)

        return FOMOSuitabilityCheck(x, bboxes, fomo_reduction)

    def __init__(self,
                 x: np.ndarray,
                 bboxes: List[List[BoundingBox]],
                 fomo_reduction: int):

        if len(bboxes) != len(x):
            raise Exception(f"mismatch between number of images X ({len(x)})"
                            f" and number of bbox entries Y ({len(bboxes)})")

        if x.dtype != np.uint8:
            x = (x * 255).astype(np.uint8)

        # Code assumes that bbounding boxes are all normalised, assert this now
        # we _could_ try to convert where required but can't be sure we know exactly
        # TODO(mat): add bool to BBLS to explicitly track this, if ever really required
        for per_image_bboxes in bboxes:
            for bb in per_image_bboxes:
                if bb.x0 > 1 or bb.x1 > 1 or bb.y0 > 1 or bb.y1 > 1:
                    raise Exception("WARNING: FOMOSuitabilityCheck expected ONLY normalised bboxes")

        self.x = x
        self.bboxes = bboxes
        self.input_hw = x.shape[1]
        self.output_hw = self.input_hw // fomo_reduction

        # calculate confusion matrices, including the collection
        # of a set of all just_background cases ( which can be filtered
        # during the most_suitable check)
        self.confusion_matrices = []
        self.just_background_idxs = set()
        for idx in range(len(self.x)):
            bboxes = self.bboxes[idx]

            if (len(bboxes) == 0):
                self.just_background_idxs.add(idx)

            # though we expect normalised bounding boxes everything we do is
            # with respect to pixel space coords, e.g. numpy indexs etc,
            # so convert now
            bboxes = [bb.project(self.input_hw, self.input_hw) for bb in bboxes]

            # convert bboxes to binary mask
            mask = util.convert_bounding_boxes_to_mask(bboxes, height_width=self.input_hw)

            # calculate centroids
            centroids = [bb.centroid().as_int() for bb in bboxes]

            # calculate statistics related to the overlap of cells with/without centroids
            # TP count, FP proportions, TN counts, FN proportions
            proportions = self.__calculate_proportions_filled(mask, centroids, grid_size=self.output_hw)

            # also collapse proportions into something akin to a confusion by summing
            # the FP and FN proportions ( i.e. so all are effectively counts now )
            confusion = np.array([[proportions['true_negatives'], sum(proportions['false_positives'])],
                                [sum(proportions['false_negatives']), proportions['true_positives']]])

            # normalise by number of labels so we don't penalise just having a lot of labels.
            if len(bboxes) > 0:
                confusion /= len(bboxes)

            self.confusion_matrices.append(confusion)

    def global_confusion_matrix(self, normalise: bool=False) -> np.ndarray:
        # calculate a global version of the confusion matrix
        # but just summing the confusion for each instance.
        global_confusion = np.zeros((2, 2), dtype=float)
        for idx in range(len(self.x)):
            global_confusion += self.confusion_matrices[idx]
        if normalise:
            global_confusion /= global_confusion.sum()
        return global_confusion

    def off_diag_score(self, idx: int, normalise: bool=False) -> float:
        # provide a score for an instance as the sum of the
        # off diagonal values of the confusion. if the
        # bboxes perfectly align with the FOMO grid this score
        # will be 0.
        confusion = self.confusion_matrices[idx]
        if normalise:
            confusion /= confusion.sum()
        return confusion[0, 1] + confusion[1, 0]

    def least_suitable(self, n: int) -> List[Tuple[int, float]]:
        # return the indexes and scores for the n least suitable instances
        idxs_scores = []
        for idx in range(len(self.x)):
            idxs_scores.append((idx, self.off_diag_score(idx)))
        arg_sorted = sorted(idxs_scores, key=lambda kv: kv[1])
        return list(reversed(arg_sorted[-n:]))

    def most_suitable(self, n: int, ignore_all_background: bool=True) -> List[Tuple[int, float]]:
        # return the indexes and scores for the n most suitable instances
        # we generally want to ignore_all_background since instances with no bounding
        # boxes automatically score perfectly ( and aren't interesting )
        # TODO(mat) perhaps it's better to have a min_labels>1 here instead of ignore_all_background ?
        idxs_scores = []
        for idx in range(len(self.x)):
            if (ignore_all_background and idx in self.just_background_idxs):
                continue
            idxs_scores.append((idx, self.off_diag_score(idx)))
        arg_sorted = sorted(idxs_scores, key=lambda kv: kv[1])
        return arg_sorted[:n]

    def img_with_debug_overlay_for(self, idx: int) -> Image:

        # load image
        if len(self.x.shape) == 4 and self.x.shape[-1] == 3:
            # RGB image
            img = Image.fromarray(self.x[idx])
        else:
            # grayscale image; X is either (B,W,H) or (B,H,W,1)
            img = Image.fromarray(self.x[idx].squeeze(), mode='L').convert('RGB')

        # overlay with some debug info
        img = self.__overlay_grid_and_boxes(img,
                                            grid_size=self.output_hw,
                                            bboxes=self.bboxes[idx])

        return img

    def __calculate_proportions_filled(self,
                                       mask: np.ndarray,
                                       centroids: List[Centroid],
                                       grid_size: int) -> Dict:
        hw = mask.shape[0]

        def cell_has_a_centroid(x1, y1, x2, y2):
            for c in centroids:
                if x1 < c.x and c.x < x2 and y1 < c.y and c.y < y2:
                    return True
            return False

        confusion = {
            'true_positives': 0,    # number of cells with centroid fully in a bounding box
            'false_positives': [],  # proportions of cells with a centroid that didn't have a bounding box fully overlapping
            'true_negatives': 0,    # number of cells without a centroid and no bounding box overlap
            'false_negatives': []   # proportions of cells without a centroid that had some bounding box overlap
        }

        cell_area = grid_size * grid_size
        for x1 in range(0, hw, grid_size):
            x2 = x1 + grid_size
            for y1 in range(0, hw, grid_size):
                y2 = y1 + grid_size
                cell = mask[x1:x2, y1:y2]
                area_filled = np.count_nonzero(cell)
                proportion = area_filled / cell_area
                has_centroid = cell_has_a_centroid(x1, y1, x2, y2)
                if has_centroid:
                    if proportion == 1:
                        # this cell has a centroid and is completely contained in
                        # a bounding box, so cell counts entirely to TPs
                        confusion['true_positives'] += 1
                    else:
                        # this cell has a centroid but isn't completely contained
                        # within a boundary. count the difference towards FPs
                        confusion['false_positives'].append(1-proportion)
                else:
                    if proportion == 0:
                        # this cell does not have a centroid and has no overlap
                        # with any boundinb boxes, so cell counts entirely
                        # to TNs
                        confusion['true_negatives'] += 1
                    else:
                        # this cell does not have a centroid but has some overlap
                        # with bounding box. count the proportion of the overlap
                        # towards FNs
                        confusion['false_negatives'].append(proportion)

        return confusion

    def __overlay_grid_and_boxes(self,
                                 img: Image,
                                 grid_size: int,
                                 bboxes: List[BoundingBox]) -> Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        assert w==h  # or just make following code more genearal

        # draw grid
        for xy in range(0, w, grid_size):
            draw.line((xy, 0, xy, h), fill='green')
            draw.line((0, xy, w, xy), fill='green')

        # draw bboxes
        for bbox in bboxes:
            bb = bbox.project(h, w)
            bb = bb.transpose_x_y()  # for application against PIL image
            # draw original true bounding box
            draw.rectangle((bb.x0, bb.y0, bb.x1, bb.y1), outline='red')
            # draw point at centroid
            c = bb.centroid().as_int()
            draw.rectangle((c.x-1, c.y-1, c.x+1, c.y+1), outline='yellow')
            # draw cell that centroid is in
            cx0 = (c.x // grid_size) * grid_size
            cy0 = (c.y // grid_size) * grid_size
            cx1 = cx0 + grid_size
            cy1 = cy0 + grid_size
            draw.rectangle((cx0, cy0, cx1, cy1), outline='yellow')

        return img

    @staticmethod
    def infer_square_shape(x: Tuple[int]) -> Tuple[int]:
        num_instances, spatial_dims = x.shape
        # try (hw, hw, 3) then (hw, hw, 1)
        for channels in [3, 1]:
            hw = int(math.sqrt(spatial_dims/channels))
            if spatial_dims == hw * hw * channels:
                return num_instances, hw, hw, channels
        raise Exception("couldn't derive square shape for", x.shape)
