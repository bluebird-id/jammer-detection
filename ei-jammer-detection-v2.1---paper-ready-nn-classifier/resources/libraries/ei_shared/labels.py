from typing import Type, Union, List

import numpy as np
import math
import json

class Centroid(object):
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx+dy*dy)

    def as_int(self):
        return Centroid(int(self.x), int(self.y), self.label)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return str({'x': self.x, 'y': self.y, 'label': self.label})

class BoundingBox(object):

    def from_x_y_h_w(x, y, h, w):
        return BoundingBox(x, y, x+w, y+h)

    def from_dict(d: dict):
        return BoundingBox(d['x0'], d['y0'], d['x1'], d['y1'])

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def close(self, other, atol) -> bool:
        return (np.isclose(self.x0, other.x0, atol=atol) and
                np.isclose(self.y0, other.y0, atol=atol) and
                np.isclose(self.x1, other.x1, atol=atol) and
                np.isclose(self.y1, other.y1, atol=atol))

    def project(self, width:int, height:int):
        return BoundingBox(self.x0 * width, self.y0 * height,
                           self.x1 * width, self.y1 * height)

    def floored(self):
        return BoundingBox(math.floor(self.x0), math.floor(self.y0),
                           math.floor(self.x1), math.floor(self.y1))

    def transpose_x_y(self):
        return BoundingBox(self.y0, self.x0, self.y1, self.x1)

    def centroid(self):
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        return Centroid(cx, cy, label=None)

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

    def update_with_overlap(self, other) -> bool:
        """ update ourselves with any overlap. return true if there was overlap """
        if (other.x0 > self.x1 or other.x1 < self.x0 or
            other.y0 > self.y1 or other.y1 < self.y0):
            return False
        if other.x0 < self.x0:
            self.x0 = other.x0
        if other.y0 < self.y0:
            self.y0 = other.y0
        if other.x1 > self.x1:
            self.x1 = other.x1
        if other.y1 > self.y1:
            self.y1 = other.y1
        return True

    def __eq__(self, other) -> bool:
        return self.close(other, atol=1e-8)

    def __iter__(self):
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1

    def __repr__(self) -> str:
        return str({'x0': self.x0, 'y0': self.y0, 'x1': self.x1, 'y1': self.y1})

class BoundingBoxLabelScore(object):

    def from_dict(d: dict):
        bbox = BoundingBox.from_dict(d['bbox'])
        return BoundingBoxLabelScore(bbox, d['label'], d['score'])

    def from_bounding_box_labels_file(fname):
        """
        Parse a bounding_box.labels file as exported from Studio and return a
        dictionary with key = source filename & value = [BoundingBoxLabelScore]
        """
        with open(fname, 'r') as f:
            labels = json.loads(f.read())
            if labels['version'] != 1:
                raise Exception(f"Unsupported file version [{labels['version']}]")
            result = {}
            for fname, bboxes in labels['boundingBoxes'].items():
                bbls = []
                for bbox in bboxes:
                    x, y = bbox['x'], bbox['y']
                    w, h = bbox['width'], bbox['height']
                    bbls.append(BoundingBoxLabelScore(
                        BoundingBox.from_x_y_h_w(x, y, h, w),
                        bbox['label']))
                result[fname] = bbls
            return result

    def __init__(self, bbox: BoundingBox, label: int, score: float=None):
        self.bbox = bbox
        self.label = label
        self.score = score

    def centroid(self):
        centroid = self.bbox.centroid()
        centroid.label = self.label
        return centroid

    def __eq__(self, other) -> bool:
        if self.score is None or other.score is None:
            score_equal = self.score == other.score
        else:
            score_equal = np.isclose(self.score, other.score)
        return (score_equal and
                self.bbox == other.bbox and
                self.label == other.label)

    def __repr__(self) -> dict:
        return str({'bbox': str(self.bbox), 'label': self.label,
                    'score': self.score })

class Labels:
    """Represents a set of labels for a classification problem"""

    def __init__(self, labels: 'list[str]'):
        if len(set(labels)) < len(labels):
            raise ValueError('No duplicates allowed in label names')
        self._labels_str = labels

    # Need to upgrade to numpy >= 1.2.0 to get proper type support
    def __getitem__(self, lookup: 'Union[int, np.integer, str]'):
        if isinstance(lookup, (int, np.integer)):
            if lookup < 0:
                raise IndexError(f'Index {lookup} is too low')
            if lookup >= len(self._labels_str):
                raise IndexError(f'Index {lookup} is too high')
            return Label(self, int(lookup), self._labels_str[lookup])
        elif isinstance(lookup, str):
            return Label(self, self._labels_str.index(lookup), lookup)
        else:
            raise IndexError(f'Index {lookup} is not in the list of labels')

    def __len__(self):
        return len(self._labels_str)

    def __iter__(self):
        for idx in range(0, len(self._labels_str)):
            yield Label(self, idx, self._labels_str[idx])


class Label:
    """Represents an individual label for a classification problem"""

    def __init__(self, labels: Labels, label_idx: int, label_str: str):
        self._labels = labels
        self._label_idx = label_idx
        self._label_str = label_str

    @property
    def idx(self):
        return self._label_idx

    @property
    def str(self) -> str:
        return self._label_str

    @property
    def all_labels(self):
        return self._labels

    def __eq__(self, other):
        if isinstance(other, Label):
            # Individual labels are only the same if they come from the same list
            if list(other.all_labels._labels_str) != list(self.all_labels._labels_str):
                raise ValueError('Cannot compare Label from different sets')
            return self._label_idx == other._label_idx
        raise TypeError('Cannot compare Label with non-labels')
