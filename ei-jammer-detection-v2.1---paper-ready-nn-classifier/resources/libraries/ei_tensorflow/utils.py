import os, json
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional

def is_y_structured(file_path):
    with open(file_path, 'rb') as f:
        first_byte = f.read(1)
        if (first_byte == b'{'):
            return True
        else:
            return False

def load_y_structured(dir_path, file_name, num_samples):
    with open(os.path.join(dir_path, file_name), 'r') as file:
        Y_structured_file = json.loads(file.read())
    if not Y_structured_file['version'] or Y_structured_file['version'] != 1:
        print('Unknown version for structured labels. Cannot continue, please contact support.')
        exit(1)

    Y_structured = Y_structured_file['samples']

    if len(Y_structured) != num_samples:
        print('Structured labels should have same length as samples array. Cannot continue, please contact support.')
        exit(1)

    return Y_structured

def load_validation_split_metadata(dir_path, file_name):
    validation_split_metadata_path = os.path.join(dir_path, file_name)
    if (not os.path.exists(validation_split_metadata_path)):
        return None

    with open(validation_split_metadata_path, 'r') as file:
        return json.loads(file.read())

def convert_box_coords(box: dict, width: int, height: int):
    # TF standard format is [y_min, x_min, y_max, x_max]
    # expressed from 0 to 1
    return [box['y'] / height,
            box['x'] / width,
            (box['y'] + box['h']) / height,
            (box['x'] + box['w']) / width]

def process_bounding_boxes(raw_boxes: list, width: int, height: int, num_classes: int):
    boxes = []
    classes = []
    for box in raw_boxes:
        coords = convert_box_coords(box, width, height)
        boxes.append(coords)
        # The model expects classes starting from 0
        # TODO: Use a more efficient way of doing one hot
        classes.append(tf.one_hot(box['label'] - 1, num_classes).numpy())

    # We have to make sure the correct shape is propagated even for lists that have zero elements
    boxes_tensor = tf.ragged.constant(boxes, inner_shape=[len(raw_boxes), 4])
    classes_tensor = tf.ragged.constant(classes, inner_shape=[len(raw_boxes), num_classes])
    return tf.ragged.stack([boxes_tensor, classes_tensor], axis=0)

def calculate_freq(interval):
    """Determines the frequency of a signal given its interval

    Args:
        interval (_type_): Interval in ms

    Returns:
        _type_: Frequency in Hz
    """
    # Determines the frequency of a signal given its interval
    freq = 1000 / interval
    if abs(freq - round(freq)) < 0.001:
        freq = round(freq)
    return freq

def can_cache_data(X_train):
    """Returns True if data will fit in cache"""
    X_train_size_bytes = X_train.size * X_train.itemsize
    max_memory_bytes = 0
    if os.environ.get("EI_MAX_MEMORY_MB"):
        max_memory_bytes = int(os.environ.get("EI_MAX_MEMORY_MB")) * 1024 * 1024

    return (X_train_size_bytes * 2) < max_memory_bytes

def scale_image(img: npt.NDArray, input_scaling: Optional[Literal['0..255', '0..1', '-1..1', 'torch']]):
    """
    Scales a numpy array representing a 3 channel image to the given input range, in place.
    """
    if len(img.shape) != 3:
        raise ValueError(f'Expected 3 image dimensions, but shape was {img.shape}')
    _, _, channels = img.shape

    # Perform scaling of the image data, if necessary for the model.
    # See keras-types.ts for the definition of these strings.
    if input_scaling == '0..255':
        # The image is always 0..255 to begin with.
        pass
    elif input_scaling is None or input_scaling == '0..1':
        # Use 0..1 as the default if nothing is specified
        img = (img / 255.0).astype(np.float32)
    elif input_scaling == '-1..1':
        img = ((img * 2) - 1).astype(np.float32)
    elif input_scaling == 'torch':
        if channels == 1:
            raise ValueError('Cannot use "torch" scaling with grayscale input')
        # Scale to 0..1
        img = (img / 255.0).astype(np.float32)
        # Normalize using ImageNet mean/std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img[:,:,0] = (img[:,:,0] - mean[0]) / std[0]
        img[:,:,1] = (img[:,:,1] - mean[1]) / std[1]
        img[:,:,2] = (img[:,:,2] - mean[2]) / std[2]
    else:
        raise ValueError(f'Unrecognized input scaling type {input_scaling}')

    return img