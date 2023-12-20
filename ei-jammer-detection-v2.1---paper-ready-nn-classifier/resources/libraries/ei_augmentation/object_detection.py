import sys
import numpy as np

albumentations_import_err = None
ALBUMENTATIONS_IMPORTED = True
try:
    from albumentations import Compose, RandomResizedCrop, Rotate, RandomBrightnessContrast, HorizontalFlip
    from albumentations import BboxParams
except ImportError as err:
    albumentations_import_err=err
    ALBUMENTATIONS_IMPORTED = False

class Augmentation(object):

    def __init__(self, width: int, height: int, num_channels: int):
        self.albumentations_error_printed = False
        self.width = width
        self.height = height
        self.num_channels = num_channels

        if not ALBUMENTATIONS_IMPORTED:
            return

        self.transform = Compose([
            RandomResizedCrop(height=self.height,
                               width=self.width,
                               scale=(0.90, 1.1)),
            Rotate(limit=20),
            RandomBrightnessContrast(brightness_limit=0.2,
                                     contrast_limit=0.2,
                                     p=0.5),
            HorizontalFlip(p=0.5),
        ], bbox_params=BboxParams(format='coco',
                                  label_fields=['class_labels']))

    def augment(self, x: np.array, bboxes_dict: dict):
        if not ALBUMENTATIONS_IMPORTED:
            if not self.albumentations_error_printed:
                print('ERROR: Could not load albumentations library. This is a known issue '
                    'on M1 Macs (#3880) and will prevent object detection augmentation from working. \n'
                    'Original error message:\n',
                    albumentations_import_err, file=sys.stderr)
                self.albumentations_error_printed = True

            return x, bboxes_dict

        # convert from studio formats to albumentations format
        # x gets proper 3d shape
        # y gets split into bboxes and labels as seperate vars
        x2 = x.reshape(self.width, self.height, self.num_channels)
        np_bboxes = []
        for bbox in bboxes_dict:
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            np_bboxes.append((x, y, w, h))
        np_bboxes = np.array(np_bboxes)
        labels = [b['label'] for b in bboxes_dict]

        # run augmentation
        transformed = self.transform(image=x2,
                                     bboxes=np_bboxes,
                                     class_labels=labels)
        augmented_x = transformed['image']
        augmented_bboxes = transformed['bboxes']
        augmented_labels = transformed['class_labels']

        # convert back from the albumentations formats to studio format
        augmented_x = augmented_x.flatten()
        augmented_bboxes_dict = []
        for a_bboxes, label in zip(augmented_bboxes, augmented_labels):
            x, y, w, h = a_bboxes
            augmented_bboxes_dict.append(
                {'label': label, 'x': int(x), 'y': int(y),
                 'w': int(w), 'h': int(h)})

        return augmented_x, augmented_bboxes_dict
