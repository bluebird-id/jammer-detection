import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Softmax, Conv2D
from tensorflow.keras.models import Model
import tempfile
import os
import shutil
import zipfile
from typing import Tuple

from ei_tensorflow.training import get_concrete_function
from ei_tensorflow.conversion import convert_float32, convert_int8_io_int8

def convert_to_tf_lite(dir_path: str, model: Model,
                       saved_model_dir: str,
                       h5_model_path: str,
                       validation_dataset: tf.data.Dataset,
                       model_filenames_float: str,
                       model_filenames_quantised_int8: str,
                       disable_per_channel: bool) -> Tuple[bytes, bytes]:
    """Convert keras model to float32 and int8 tf lite models.

    Args:
        dir_path: base path for saving converted models.
        model: keras model to convert.
        saved_model_dir: path to save base keras model.
        h5_model_path: path to save h5 exported model.
        validation_dataset: representative dataset used for int8 conversion.
        model_filenames_float: filename, under dir_path, to write float model.
        model_filenames_quantised_int8: filename, under dir_path, to write int8
            model.

    Returns:
        (tflite_model, tflite_quant_model) both as bytes.
    """

    # derive model input size from first layer
    # TODO(mat) remove restriction that everything is square
    model_input_shape = model.layers[0].input.shape
    _batch, width, height, input_num_channels = model_input_shape
    if width != height:
        raise Exception(f"Only square inputs are supported; not {model_input_shape}")
    input_width_height = width

    # use model layer to derive output size as well as number of classes
    model_output_shape = model.layers[-1].output.shape
    _batch, width, height, num_classes = model_output_shape
    if width != height:
        raise Exception(f"Only square outputs are supported; not {model_output_shape}")
    output_width_height = width

    # Legacy expert mode models (before #4409) do not include
    # the final softmax layer, so we need to add it here.
    #
    # The last layers of legacy expert models were Conv2D -> Conv2D -> Softmax.
    # We explicitly look for these layers
    # to allow for last layers as follows: Reshape -> Softmax -> Reshape
    # required e.g. Tensai Flow Library
    if not isinstance(model.layers[-1], Softmax) and isinstance(model.layers[-2], Conv2D) and isinstance(model.layers[-3], Conv2D):
        softmax_layer = Softmax()(model.layers[-1].output)
        model = Model(model.input, softmax_layer)

    # Save the model to disk and zip it for download.
    # Save both TF savedmodel and keras h5 for maximum compatibility.
    # TODO(mat) this is cut-n-paste from ei_tensorflow.conversion.convert_to_tf_lite
    #    and isn't done for non FOMO object detection models (object_detection.convert_to_tf_lite)
    saved_model_path = os.path.join(dir_path, saved_model_dir)
    model.save(saved_model_path, save_format='tf')
    shutil.make_archive(os.path.join(dir_path, saved_model_dir),
                        'zip', root_dir=dir_path,
                        base_dir=saved_model_dir)
    h5_path = os.path.join(dir_path, h5_model_path)
    model.save(h5_path, save_format='h5')
    with zipfile.ZipFile(h5_path + '.zip', "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(h5_path, os.path.basename(h5_path))
    os.remove(h5_path)

    # since get concrete needs an explicit serving_default target we need to
    # save/load
    tmp_path = tempfile.gettempdir()
    model.save(tmp_path)
    def weighted_xent(y_true, y_pred_logits):
        # this model uses a custom loss function which is not serialised by default
        # easiest workaround is to provide a noop implementation for keras to use
        # in loading. we aren't continuing training so this workaround is fine.
        raise Exception("didnt expect weighted_xent to be called during tf lite conversion")
    model = keras.models.load_model(filepath=tmp_path,
                                    custom_objects={'weighted_xent': weighted_xent})

    # get concrete function
    input_shape_without_batch_dim = model.input.shape[1:]
    concrete_func = get_concrete_function(model,
                                          input_shape_without_batch_dim)

    # convert float32 model
    tflite_model = convert_float32(concrete_func, model, dir_path, model_filenames_float)

    # convert quanitised model
    def representative_dataset_gen():
        # note: needs to explicitly be a batch of a single example at a time,
        # returned in an array
        for data, _ in validation_dataset.batch(1):
            yield [data]

    tflite_quant_model = convert_int8_io_int8(
        concrete_func, model, representative_dataset_gen,
        dir_path, model_filenames_quantised_int8, disable_per_channel)

    return tflite_model, tflite_quant_model
