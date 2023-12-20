import matplotlib
import matplotlib.pyplot as plt

import os
import shutil
import random
import io
import glob
import scipy.misc
import time
import numpy as np
from six import BytesIO

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

from object_detection.export_tflite_graph_lib_tf2 import export_tflite_model
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from ei_tensorflow.training import get_friendly_time, print_training_time_exceeded, check_gpu_time_exceeded

from ei_tensorflow.conversion import run_converter

MAX_TRAINING_TIME_S = 24 * 60 * 60
MAX_GPU_TIME_S = 1500 * 60
IS_ENTERPRISE_PROJECT = False

def set_limits(max_training_time_s, max_gpu_time_s, is_enterprise_project):
    global MAX_TRAINING_TIME_S, MAX_GPU_TIME_S, IS_ENTERPRISE_PROJECT

    MAX_TRAINING_TIME_S = max_training_time_s
    MAX_GPU_TIME_S = max_gpu_time_s
    IS_ENTERPRISE_PROJECT = is_enterprise_project

def train(num_classes, learning_rate, num_epochs, train_dataset, validation_dataset):
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)

    object_detection_dir = './object_detection' if os.path.exists('./object_detection') else '/app/keras/models/research/object_detection'

    pipeline_config = os.path.join(object_detection_dir, 'configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config')
    checkpoint_path = os.path.join(object_detection_dir, 'test_data/checkpoint/ckpt-0')

    # This will be where we save checkpoint & config for TFLite conversion later.
    output_directory = 'output/'
    output_checkpoint_dir = os.path.join(output_directory, 'checkpoint')

    # Load pipeline config and build a detection model.
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    # Save new pipeline config
    pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
    config_util.save_pipeline_config(pipeline_proto, output_directory)

    # Set up object-based checkpoint restore --- SSD has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
    fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=detection_model._feature_extractor,
            _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # To save checkpoint for TFLite conversion.
    exported_ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_manager = tf.train.CheckpointManager(
        exported_ckpt, output_checkpoint_dir, max_to_keep=1)

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 320, 320, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Finished restoring weights')

    # This was in the tutorial code and prints a warning. It doesn't seem to be
    # necessary but I'm leaving it here for a while to make debugging easier just in case.
    # tf.keras.backend.set_learning_phase(True)

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    validation_fn = get_model_validation_function(detection_model)

    run_loop(num_epochs, train_dataset, num_classes, train_step_fn, validation_dataset, validation_fn,
        MAX_TRAINING_TIME_S, MAX_GPU_TIME_S, IS_ENTERPRISE_PROJECT)

    print('Finished fine tuning')

    ckpt_manager.save()
    print('Checkpoint saved')

    return detection_model

def run_loop(num_epochs, train_dataset, num_classes, train_step_fn, validation_dataset, validation_fn,
             max_training_time_s, max_gpu_time_s, is_enterprise_project):
    print('Fine tuning...', flush=True)

    epoch_0_begin = time.time()
    epoch_1_begin = time.time()
    epoch_0_time_s = 0

    for idx in range(num_epochs):
        epoch_begin = time.time()

        if (idx == 0):
            epoch_0_begin = time.time()
        if (idx == 1):
            epoch_1_begin = time.time()

        training_loss = None
        # Loop through all batches.
        for batch in train_dataset:
            image_tensors = batch[0]
            # We have to reshape the boxes and classes since they arrive as ragged tensors
            gt_boxes_list = [boxes.to_tensor(shape=[None, 4]) for boxes in batch[1][0]]
            gt_classes_list = [classes.to_tensor(shape=[None, num_classes]) for classes in batch[1][1]]
            # Pass batch into training function
            training_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        # Perform validation over the whole validation set each epoch
        val_loss = 0
        batch_count = 0
        for batch in validation_dataset:
            image_tensors = batch[0]
            gt_boxes_list = [boxes.to_tensor(shape=[None, 4]) for boxes in batch[1][0]]
            gt_classes_list = [classes.to_tensor(shape=[None, num_classes]) for classes in batch[1][1]]
            val_loss += validation_fn(image_tensors, gt_boxes_list, gt_classes_list)
            batch_count += 1

        val_loss /= batch_count

        # on both epoch 0 and epoch 1 we want to estimate training time
        # if either is above the training time limit, then we exit
        if (idx == 0 or idx == 1):
            time_per_epoch_ms = 0
            if (idx == 0):
                time_per_epoch_ms = float(time.time() - epoch_0_begin) * 1000

                epoch_0_time_s = time_per_epoch_ms / 1000
            elif (idx == 1):
                time_per_epoch_ms = float(time.time() - epoch_1_begin) * 1000

            total_time = time_per_epoch_ms * num_epochs / 1000

            if (idx == 0):
                # so for SSD models we see ~7x longer time for the first epoch
                # we can safely adjust this down here, another check is made at the
                # end of the second epoch anyway
                total_time = ((time_per_epoch_ms / 7) * (num_epochs - 1) / 1000) + epoch_0_time_s
            else:
                # for the total time, adjust the total time with the actual time for first epoch
                total_time = (time_per_epoch_ms * (num_epochs - 1) / 1000) + epoch_0_time_s

            # uncomment this to debug the training time algo:
            # print('Epoch', idx, '- time for this epoch: ' + get_friendly_time(time_per_epoch_ms / 1000) +
            #     ', estimated training time:', get_friendly_time(total_time))

            if (total_time > max_training_time_s * 1.2):
                print_training_time_exceeded(is_enterprise_project, max_training_time_s, total_time)
                exit(1)
            check_gpu_time_exceeded(max_gpu_time_s, total_time)

        if idx % 1 == 0:
            print('Epoch ' + str(idx + 1) + ' of ' + str(num_epochs)
            + ', loss=' +  str(training_loss.numpy())
            + ', val_loss=' + str(val_loss.numpy()), flush=True)

    # uncomment this to debug the training time algo:
    # print('Total training time:', get_friendly_time(time.time() - epoch_0_begin))

# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the
    # function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors,
                        groundtruth_boxes_list,
                        groundtruth_classes_list):
        """A single training iteration.

        Args:
        image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 320x320.
        groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
        groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
        A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(len(image_tensors) * [[320, 320, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)

        # The images each have a pointless batch dimension of 1, so do a reshape
        # to remove this from the result of concatenation
        concatted = tf.reshape(tf.concat(image_tensors, axis=0), (len(image_tensors), 320, 320, 3))

        # Only watch the variables we care about to avoid excessive memory use
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for var in vars_to_fine_tune:
                tape.watch(var)
            prediction_dict = model.predict(concatted, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

        gradients = tape.gradient(total_loss, vars_to_fine_tune)
        optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))

        return total_loss

    return train_step_fn

def get_model_validation_function(model):
    @tf.function
    def validation_function(image_tensors,
                            groundtruth_boxes_list,
                            groundtruth_classes_list,
                            use_mean_loss=True):

        shapes = tf.constant(len(image_tensors) * [[320, 320, 3]], dtype=tf.int32)
        model.provide_groundtruth(
            groundtruth_boxes_list=groundtruth_boxes_list,
            groundtruth_classes_list=groundtruth_classes_list)
        # The images each have a pointless batch dimension of 1, so do a reshape
        # to remove this from the result of concatenation
        concatted = tf.reshape(tf.concat(image_tensors, axis=0), (len(image_tensors), 320, 320, 3))
        prediction_dict = model.predict(concatted, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        return total_loss

    return validation_function

def get_saved_model(dir_path, output_name):
    print('Creating SavedModel for conversion...', flush=True)

    # Hide debug logs from one of the library's dependencies
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

    with tf.io.gfile.GFile('output/pipeline.config', 'r') as f:
        text_format.Parse(f.read(), pipeline_config)

    # TODO: Different number of max detections?
    max_detections = 10
    export_tflite_model(
        pipeline_config, 'output/checkpoint', '.',
        max_detections, False,
        False, None)

    # Zip the saved_model for download. The zip should end up in 'dir_path'
    # even though the saved_model is in the cwd
    shutil.make_archive(os.path.join(dir_path, output_name),
                        'zip', root_dir='.',
                        base_dir='saved_model')

    print('Finished creating SavedModel', flush=True)

    print('Loading for conversion...', flush=True)
    saved_model = tf.saved_model.load('./saved_model')
    return saved_model

def representative_dataset_generator(validation_dataset):
    def gen():
        for data_batch, _ in validation_dataset.take(-1):
            for data in data_batch:
                yield [data]
    return gen

def convert_int8_io_mixed(dataset_generator, dir_path, filename):
    """
    Used to convert TensorFlow object detection models, which have a custom op as their tail that can't be
    quantized to int8.
    """
    try:
        print('Converting TensorFlow Lite int8 quantized model with int8 input and float32 output...', flush=True)
        # This is hardcoded
        converter_quantize = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
        converter_quantize.allow_custom_ops = True
        converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quantize.representative_dataset = dataset_generator
        # Force the input to be int8
        converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_quantize.inference_input_type = tf.int8
        tflite_quant_model = run_converter(converter_quantize)
        open(os.path.join(dir_path, filename), 'wb').write(tflite_quant_model)
        return tflite_quant_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite int8 quantized model:')
        print(err)

def convert_float32(dir_path, filename):
    try:
        print('Converting TensorFlow Lite float32 model...', flush=True)
        converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model')
        converter.allow_custom_ops = True
        tflite_model = run_converter(converter)
        open(os.path.join(dir_path, filename), 'wb').write(tflite_model)
        return tflite_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite float32 model:')
        print(err)

def convert_to_tf_lite(dir_path, saved_model_dir, validation_dataset,
                       model_filenames_float, model_filenames_quantised_int8):
    _tf_saved_model = get_saved_model(dir_path, saved_model_dir)
    dataset_generator = representative_dataset_generator(validation_dataset)
    tflite_model = convert_float32(dir_path, model_filenames_float)
    tflite_quant_model = convert_int8_io_mixed(dataset_generator, dir_path,
                                               model_filenames_quantised_int8)
    return tflite_model, tflite_quant_model
