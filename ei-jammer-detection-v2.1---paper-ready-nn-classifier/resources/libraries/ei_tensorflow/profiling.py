
from __future__ import print_function
import json
import time
import traceback
import os
import numpy as np
import tensorflow as tf
import json, datetime, time, traceback
import os, shutil, operator, functools, time, subprocess, math
from typing import Optional
import ei_tensorflow.inference
import ei_tensorflow.brainchip.model
from concurrent.futures import ThreadPoolExecutor
import ei_tensorflow.utils
from ei_shared.types import ClassificationMode, ObjectDetectionDetails
import ei_tensorflow.tao_inference.retinanet

from ei_tensorflow.constrained_object_detection.util import batch_convert_segmentation_map_to_object_detection_prediction
from ei_tensorflow.constrained_object_detection.metrics import dataset_match_by_near_centroids

from ei_sklearn.metrics import calculate_regression_metrics
from ei_sklearn.metrics import calculate_classification_metrics
from ei_sklearn.metrics import calculate_object_detection_metrics
from ei_coco.metrics import calculate_coco_metrics
from ei_sklearn.metrics import calculate_fomo_metrics

def ei_log(msg: str):
    print("EI_LOG_LEVEL=debug", msg)

def tflite_predict(model, validation_dataset, dataset_length, item_feature_axes: Optional[list]=None):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        if item_feature_axes:
            item_as_tensor = np.take(item_as_tensor, item_feature_axes)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        scores = ei_tensorflow.inference.process_output(output_details, output)
        pred_y.append(scores)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def tflite_predict_object_detection(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            interpreter.invoke()
            rect_label_scores = ei_tensorflow.inference.process_output_object_detection(output_details, interpreter)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

# Y_test is required to generate anchors for YOLOv2 output decoding
def tflite_predict_yolov2(model, validation_dataset, Y_test, dataset_length, num_classes, output_directory):
    import pickle

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    with open(os.path.join(output_directory, "akida_yolov2_anchors.pkl"), 'rb') as handle:
        anchors = pickle.load(handle)

    last_log = time.time()
    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            h, w, c = output.shape
            output = output.reshape((h, w, len(anchors), 4 + 1 + num_classes))
            rect_label_scores = ei_tensorflow.brainchip.model.process_output_yolov2(output, (width, height), num_classes, anchors)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    result = np.array(pred_y, dtype=object)
    return result

def tflite_predict_yolov5(model, version, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            _batch, width, height, _channels = input_details[0]['shape']
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = np.array(ei_tensorflow.inference.process_output(output_details, output))
            # expects to have batch dim here, eg (1, 5376, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolov5(output, (width, height),
                version)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolox(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)

            _batch, width, height, _channels = input_details[0]['shape']
            if width != height:
                raise Exception(f"expected square input, got {input_details[0]['shape']}")

            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = np.array(ei_tensorflow.inference.process_output(output_details, output))
            # expects to have batch dim here, eg (1, 5376, 6)
            # if not, then add batch dim
            if len(output.shape) == 2:
                output = np.expand_dims(output, axis=0)
            rect_label_scores = ei_tensorflow.inference.process_output_yolox(output, img_size=width)
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_yolov7(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
            item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            output = ei_tensorflow.inference.process_output(output_details, output)
            rect_label_scores = ei_tensorflow.inference.process_output_yolov7(output,
                width=input_details[0]['shape'][1], height=input_details[0]['shape'][2])
            pred_y.append(rect_label_scores)
            # Print an update at least every 10 seconds
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    # Must specify dtype=object since it is a ragged array
    return np.array(pred_y, dtype=object)

def tflite_predict_segmentation(model, validation_dataset, dataset_length):
    """Runs a TensorFlow Lite model across a set of inputs"""

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    last_log = time.time()

    y_pred = []
    for item, _ in validation_dataset.take(-1):
        item_as_tensor = ei_tensorflow.inference.process_input(input_details, item)
        item_as_tensor = tf.reshape(item_as_tensor, input_details[0]['shape'])
        interpreter.set_tensor(input_details[0]['index'], item_as_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        output = ei_tensorflow.inference.process_output(output_details, output)
        y_pred.append(output)
        # Print an update at least every 10 seconds
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(y_pred) - 1))), flush=True)
            last_log = current_time

    y_pred = np.stack(y_pred)

    return y_pred

def get_tensor_details(tensor):
    """Obtains the quantization parameters for a given tensor"""
    details = {
        'dataType': None,
        'name': tensor['name'],
        'shape': tensor['shape'].tolist(),
        'quantizationScale': None,
        'quantizationZeroPoint': None
    }
    if tensor['dtype'] is np.int8:
        details['dataType'] = 'int8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.uint8:
        details['dataType'] = 'uint8'
        details['quantizationScale'] = tensor['quantization'][0]
        details['quantizationZeroPoint'] = tensor['quantization'][1]
    elif tensor['dtype'] is np.float32:
        details['dataType'] = 'float32'
    else:
        raise Exception('Model tensor has an unknown datatype, ', tensor['dtype'])

    return details


def get_io_details(model, model_type):
    """Gets the input and output datatype and quantization details for a model"""
    interpreter = tf.lite.Interpreter(model_content=model)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = list(map(get_tensor_details, input_details))
    outputs = list(map(get_tensor_details, output_details))

    return {
        'modelType': model_type,
        'inputs': inputs,
        'outputs': outputs
    }

def make_predictions(mode: ClassificationMode, model, x_dataset, y, num_classes,
                     output_directory: str, item_feature_axes: Optional[list]=None,
                     objdet_details: Optional[ObjectDetectionDetails]=None):
    if x_dataset is None:
        return None
    else:
        return make_predictions_tflite(mode, model, x_dataset, y, num_classes,
                                       output_directory, item_feature_axes,
                                       objdet_details)

def make_predictions_tflite(mode: ClassificationMode, model, x_dataset, y, num_classes,
                            output_directory, item_feature_axes: Optional[list]=None,
                            objdet_details: Optional[ObjectDetectionDetails]=None):
    if mode == 'object-detection':
        if objdet_details is None:
            raise ValueError('objdet_details must be provided for object-detection mode')
        if objdet_details.last_layer == 'mobilenet-ssd':
            return tflite_predict_object_detection(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov5':
            return tflite_predict_yolov5(model, 6, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov2-akida':
            return tflite_predict_yolov2(model, x_dataset, y, len(y), num_classes, output_directory)
        elif objdet_details.last_layer == 'yolov5v5-drpai':
            return tflite_predict_yolov5(model, 5, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolox':
            return tflite_predict_yolox(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'yolov7':
            return tflite_predict_yolov7(model, x_dataset, len(y))
        elif objdet_details.last_layer == 'tao-retinanet':
            return ei_tensorflow.tao_inference.retinanet.tflite_predict(model, x_dataset, len(y), objdet_details)
        elif objdet_details.last_layer == 'fomo':
            return tflite_predict_segmentation(model, x_dataset, len(y))
        else:
            raise Exception(f'Expecting a supported object detection last layer (got {objdet_details.last_layer})')
    elif mode == 'visual-anomaly':
        raise Exception('Expecting a supported mode to make predictions (visual-anomaly is not)')
    else:
        return tflite_predict(model, x_dataset, len(y), item_feature_axes)

def profile_model(model_type, model, model_file, validation_dataset, Y_test, X_samples, Y_samples,
                         has_samples, memory, mode: ClassificationMode, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                         num_classes, train_dataset, Y_train, test_dataset, Y_real_test,
                         akida_model_path,
                         item_feature_axes: list,
                         async_memory_profiling: bool,
                         objdet_details: Optional[ObjectDetectionDetails]):

    """Calculates performance statistics for a TensorFlow Lite model"""
    matrix_train=None
    matrix_test=None
    report_train=None
    report_test=None

    # default values that are overriden based on mode
    matrix = []
    report = {}
    accuracy = 0
    prediction_samples = []
    loss = 0

    model_path = model_file if model_file is not None else akida_model_path
    output_directory = os.path.dirname(os.path.realpath(model_path))

    if mode != 'visual-anomaly':
        if akida_model_path:
            # TODO: refactor akida make_predictions in the same way we have
            #       below making three calls, one per split
            prediction, prediction_train, prediction_test = ei_tensorflow.brainchip.model.make_predictions(
                mode, akida_model_path, validation_dataset, Y_test,
                train_dataset, Y_train, test_dataset, Y_real_test,
                num_classes, output_directory,
                objdet_details=objdet_details)

            # akida returns logits so apply softmax for y_pred_prob metrics
            from scipy.special import softmax
            prediction = softmax(prediction, axis=-1)
            if prediction_train is not None:
                prediction_train = softmax(prediction_train, axis=-1)
            if prediction_test is not None:
                prediction_test = softmax(prediction_test, axis=-1)

        else:
            prediction = make_predictions(
                mode, model, validation_dataset, Y_test,
                num_classes, output_directory, item_feature_axes,
                objdet_details=objdet_details)
            prediction_train = make_predictions(
                mode, model, train_dataset, Y_train,
                num_classes, output_directory, item_feature_axes,
                objdet_details=objdet_details)
            prediction_test = make_predictions(
                mode, model, test_dataset, Y_real_test,
                num_classes, output_directory, item_feature_axes,
                objdet_details=objdet_details)

    if mode == 'classification':
        metrics = calculate_classification_metrics(
            y_true_one_hot=Y_test,
            y_pred_probs=prediction,
            num_classes=num_classes)
        ei_log(f"calculate_classification_metrics {json.dumps(metrics)}")

        matrix = metrics['confusion_matrix']
        report = metrics['classification_report']
        accuracy = metrics['classification_report']['accuracy']
        loss = metrics['loss']

        if prediction_train is not None:
            metrics = calculate_classification_metrics(
                prediction_train, Y_train, num_classes)
            matrix_train = metrics['confusion_matrix']
            report_train = metrics['classification_report']

        if prediction_test is not None:
            metrics = calculate_classification_metrics(
                prediction_test, Y_real_test, num_classes)
            matrix_test = metrics['confusion_matrix']
            report_test = metrics['classification_report']

        # TODO: move feature explorer code here and for regression into own helper
        try:
            # Make predictions for feature explorer
            if has_samples:
                if model:
                    feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples), item_feature_axes)
                elif akida_model_path:
                    feature_explorer_predictions = ei_tensorflow.brainchip.model.predict(akida_model_path, X_samples, len(Y_samples))
                else:
                    raise Exception('Expecting either a Keras model or an Akida model')

                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, np.array([feature_explorer_predictions.argmax(axis=1) + 1]).T), axis=1).tolist()
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)

    elif mode == 'regression':

        metrics = calculate_regression_metrics(
            y_true=Y_test,
            y_pred=prediction[:,0]
        )
        ei_log(f"calculate_regression_metrics {json.dumps(metrics)}")

        loss = metrics['mean_squared_error']

        try:
            # Make predictions for feature explorer
            if has_samples:
                feature_explorer_predictions = tflite_predict(model, X_samples, len(Y_samples), item_feature_axes)
                # Store each prediction with the original sample for the feature explorer
                prediction_samples = np.concatenate((Y_samples, feature_explorer_predictions), axis=1).tolist()
        except Exception as e:
            print('Failed to generate feature explorer', e, flush=True)

    elif mode == 'object-detection':
        if objdet_details is None:
            raise ValueError('objdet_details must be provided for object-detection mode')
        if objdet_details.last_layer != 'fomo':
            y_true_bbox_labels = []
            for sample in validation_dataset.take(-1).unbatch():
                y_true_bbox_labels.append(sample)

            metrics = calculate_object_detection_metrics(
                y_true_bbox_labels=y_true_bbox_labels,
                y_pred_bbox_labels=prediction,
                num_classes=num_classes
            )
            ei_log(f"calculate_object_detection_metrics {json.dumps(metrics)}")

            coco_metrics = calculate_coco_metrics(
                studio_y_true_bbox_labels=y_true_bbox_labels,
                studio_y_pred_bbox_labels=prediction,
                num_classes=num_classes
            )
            ei_log(f"calculate_coco_metrics {json.dumps(coco_metrics)}")

            accuracy = metrics['coco_map']

        else:
            _batch, width, height, num_classes = prediction.shape
            if width != height:
                raise Exception("Expected segmentation output to be square, not",
                                prediction.shape)
            output_width_height = width

            # y_true has already been extracted during tflite_predict_segmentation
            # and has labels including implicit background class = 0

            # TODO(mat): what should minimum_confidence_rating be here?
            y_pred = batch_convert_segmentation_map_to_object_detection_prediction(
                prediction, minimum_confidence_rating=0.5, fuse=True)

            # do alignment by centroids. this results in a flatten list of int
            # labels that is suitable for confusion matrix calculations.
            y_true_labels, y_pred_labels = dataset_match_by_near_centroids(
                # batch the data since the function expects it
                validation_dataset.batch(32, drop_remainder=False), y_pred, output_width_height)

            metrics = calculate_fomo_metrics(y_true_labels, y_pred_labels, num_classes)
            ei_log(f"calculate_fomo_metrics {json.dumps(metrics)}")

            matrix = metrics['confusion_matrix']
            report = metrics['classification_report']
            accuracy = metrics['non_background']['f1']

    elif mode == 'anomaly-gmm' or mode == 'visual-anomaly':
        # by definition we don't have any anomalies in the training dataset
        # so we don't calculate these metrics
        pass

    model_size = 0
    if model:
        model_size = len(model)

    if akida_model_path:
        is_supported_on_mcu = False
        mcu_support_error = "Akida models run only on Linux boards with AKD1000"
    else:
        is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(model_file, log_messages=False)

    memory_async = None
    if (is_supported_on_mcu):
        if (not memory):
            # If this is true (passed in from studio/server/training/train-templates/profile.ts)
            # then we will kick off a separate Docker container to calculate RAM/ROM.
            # Post-training the metadata is read (in studio/server/training/learn-block-keras.ts)
            # and any metrics that have `memoryAsync` will fire off a separate job (see handleAsyncMemory).
            # After the async memory is completed, the async memory job will overwrite the `memory` section
            # of the metadata (so once this job is finished, the metadata looks 100% the same as when you
            # do in-process memory profiling.
            # We can't always do this (at the moment), as e.g. the EON Tuner expects memory to be available
            # synchronous.
            if async_memory_profiling:
                memory_async = {
                    'type': 'requires-profiling',
                }
            else:
                memory = calculate_memory(model_file, model_type, prepare_model_tflite_script, prepare_model_tflite_eon_script)
    else:
        memory = {}
        memory['tflite'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }
        memory['eon'] = {
            'ram': 0,
            'rom': model_size,
            'arenaSize': 0,
            'modelSize': model_size
        }

    return {
        'type': model_type,
        'loss': loss,
        'accuracy': accuracy,
        'accuracyTrain': report_train['accuracy'] if not report_train is None else None,
        'accuracyTest': report_test['accuracy'] if not report_test is None else None,
        'confusionMatrix': matrix,
        'confusionMatrixTrain': matrix_train if not matrix_train is None else None,
        'confusionMatrixTest': matrix_test if not matrix_test is None else None,
        'report': report,
        'reportTrain': report_train,
        'reportTest': report_test,
        'size': model_size,
        'estimatedMACCs': None,
        'memory': memory,
        'memoryAsync': memory_async,
        'predictions': prediction_samples,
        'isSupportedOnMcu': is_supported_on_mcu,
        'mcuSupportError': mcu_support_error,
    }

def run_tasks_in_parallel(tasks, parallel_count):
    res = []
    with ThreadPoolExecutor(parallel_count) as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            res.append(running_task.result())
    return res

def calculate_memory(model_file, model_type, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                     calculate_non_cmsis=False):
    model_size = os.stat(model_file).st_size

    parallel_count = 4
    if (model_size > 1 * 1024 * 1024):
        parallel_count = 1

    # Some models don't have the scripts (e.g. akida) so skip this step
    if prepare_model_tflite_script or prepare_model_tflite_eon_script:
        memory = {}

        def calc_memory(id, title, is_eon, is_non_cmsis):
            try:
                print('Profiling ' + model_type + ' model (' + title + ')...', flush=True)

                benchmark_folder = f'/app/benchmark-{id}'
                script = f'{benchmark_folder}/prepare_tflite_{id}.sh'
                if (is_eon):
                    if (is_non_cmsis):
                        script = f'{benchmark_folder}/prepare_eon_cmsisnn_disabled_{id}.sh'
                    else:
                        script = f'{benchmark_folder}/prepare_eon_{id}.sh'

                out_folder = f'{benchmark_folder}/tflite-model'

                # create prep scripts
                if is_eon:
                    if is_non_cmsis:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=False, out_folder=out_folder))
                    else:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=True, out_folder=out_folder))
                else:
                    with open(script, 'w') as f:
                        f.write(prepare_model_tflite_script(model_file, out_folder=out_folder))

                args = [
                    f'{benchmark_folder}/benchmark.sh',
                    '--tflite-type', model_type,
                    '--tflite-file', model_file
                ]
                if is_eon:
                    args.append('--eon')
                if is_non_cmsis:
                    args.append('--disable-cmsis-nn')

                if os.path.exists(f'{benchmark_folder}/tflite-model'):
                    shutil.rmtree(f'{benchmark_folder}/tflite-model')
                subprocess.check_output(['sh', script]).decode("utf-8")
                tflite_output = json.loads(subprocess.check_output(args).decode("utf-8"))
                if os.getenv('K8S_ENVIRONMENT') == 'staging' or os.getenv('K8S_ENVIRONMENT') == 'test':
                    print(tflite_output['logLines'])

                if is_eon:
                    # eon is always correct in memory
                    return { 'id': id, 'output': tflite_output }
                else:
                    # add fudge factor since the target architecture is different
                    # (q: can this go since the changes in https://github.com/edgeimpulse/edgeimpulse/pull/6268)
                    old_arena_size = tflite_output['arenaSize']
                    if "anomaly" in model_file:
                        fudge_factor = 0.25
                    else:
                        fudge_factor = 0.2
                    extra_arena_size = int(math.floor((math.ceil(old_arena_size) * fudge_factor) + 1024))

                    tflite_output['ram'] = tflite_output['ram'] + extra_arena_size
                    tflite_output['arenaSize'] = tflite_output['arenaSize'] + extra_arena_size

                    return { 'id': id, 'output': tflite_output }
            except Exception as err:
                print('WARN: Failed to get memory (' + title + '): ', end='')
                print(err, flush=True)
                return { 'id': id, 'output': None }

        task_list = []

        if prepare_model_tflite_script:
            task_list.append(lambda: calc_memory(id=1, title='TensorFlow Lite Micro', is_eon=False, is_non_cmsis=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=2, title='TensorFlow Lite Micro, HW optimizations disabled', is_eon=False, is_non_cmsis=True))
        if prepare_model_tflite_eon_script:
            task_list.append(lambda: calc_memory(id=3, title='EON', is_eon=True, is_non_cmsis=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=4, title='EON, HW optimizations disabled', is_eon=True, is_non_cmsis=True))

        results = run_tasks_in_parallel(task_list, parallel_count)
        for r in results:
            if (r['id'] == 1):
                memory['tflite'] = r['output']
            elif (r['id'] == 2):
                memory['tflite_cmsis_nn_disabled'] = r['output']
            elif (r['id'] == 3):
                memory['eon'] = r['output']
            elif (r['id'] == 4):
                memory['eon_cmsis_nn_disabled'] = r['output']

    else:
        memory = None

    return memory

# Useful reference: https://machinethink.net/blog/how-fast-is-my-model/
def estimate_maccs_for_layer(layer):
    """Estimate the number of multiply-accumulates in a given Keras layer."""
    """Better than flops because there's hardware support for maccs."""
    if isinstance(layer, tf.keras.layers.Dense):
        # Ignore the batch dimension
        input_count = functools.reduce(operator.mul, layer.input.shape[1:], 1)
        return input_count * layer.units

    if (isinstance(layer, tf.keras.layers.Conv1D)
        or isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.Conv3D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        # The channel is either at the start or the end of the shape (ignoring)
        # the batch dimension
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
        else:
            input_channels = layer.input.shape[-1]
        # Ignore the batch dimension but include the channels
        output_size = functools.reduce(operator.mul, layer.output.shape[1:])
        return kernel_size * input_channels * output_size

    if (isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.SeparableConv1D)
        or isinstance(layer, tf.keras.layers.DepthwiseConv2D)):
        kernel_size = functools.reduce(operator.mul, layer.kernel_size)
        if layer.data_format == 'channels_first':
            input_channels = layer.input.shape[1]
            output_channels = layer.output.shape[1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[2:])
        else:
            input_channels = layer.input.shape[-1]
            output_channels = layer.output.shape[-1]
            # Unlike regular conv, don't include the channels
            output_size = functools.reduce(operator.mul, layer.output.shape[1:-1])
        # Calculate the MACCs for depthwise and pointwise steps
        depthwise_count = kernel_size * input_channels * output_size
        # If this is just a depthwise conv, we can return early
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return depthwise_count
        # Otherwise, calculate MACCs for the pointwise step and add them
        pointwise_count = input_channels * output_size * output_channels
        return depthwise_count + pointwise_count

    if isinstance(layer, tf.keras.Model):
        return estimate_maccs_for_model(layer)

    # For other layers just return 0. These are mostly stuff that doesn't involve MACCs
    # or stuff that isn't supported by TF Lite for Microcontrollers yet.
    return 0

def estimate_maccs_for_model(keras_model):
    maccs = 0

    # e.g. non-Keras saved model
    if not hasattr(keras_model, 'layers'):
        return maccs

    for layer in keras_model.layers:
        try:
            layer_maccs = estimate_maccs_for_layer(layer)
            maccs += layer_maccs
        except Exception as err:
            print('Error while estimating maccs for layer', flush=True)
            print(err, flush=True)
    return maccs

def describe_layers(keras_model):
    layers = []

    # e.g. non-Keras saved model
    if not hasattr(keras_model, 'layers'):
        return layers

    for l in range(len(keras_model.layers)):
        layer = keras_model.layers[l]
        input = layer.input
        if isinstance(input, list):
            input = input[0]
        layers.append({
            'input': {
                'shape': input.shape[1],
                'name': input.name,
                'type': str(input.dtype)
            },
            'output': {
                'shape': layer.output.shape[1],
                'name': layer.output.name,
                'type': str(layer.output.dtype)
            }
        })

    return layers


def get_recommended_model_type(float32_perf, int8_perf):
    # For now, always recommend int8 if available
    if int8_perf:
        return 'int8'
    else:
        return 'float32'

def get_model_metadata(keras_model, validation_dataset, Y_test, X_samples, Y_samples, has_samples,
                       class_names, curr_metadata, mode: ClassificationMode, prepare_model_tflite_script,
                       prepare_model_tflite_eon_script, model_float32=None, model_int8=None,
                       file_float32=None, file_int8=None, file_akida=None,
                       train_dataset=None, Y_train=None, test_dataset=None, Y_real_test=None,
                       async_memory_profiling=False, objdet_details: Optional[ObjectDetectionDetails]=None):

    metadata = {
        'metadataVersion': 5,
        'created': datetime.datetime.now().isoformat(),
        'classNames': class_names,
        'availableModelTypes': [],
        'recommendedModelType': '',
        'modelValidationMetrics': [],
        'modelIODetails': [],
        'mode': mode,
        'kerasJSON': None,
        'performance': None,
        'objectDetectionLastLayer': objdet_details.last_layer if objdet_details else None,
        'taoNMSAttributes': objdet_details.tao_nms_attributes if objdet_details else None
    }

    # keep metadata from anomaly (gmm) training
    item_feature_axes = None
    if (
            curr_metadata and
            'mean' in curr_metadata and
            'scale' in curr_metadata and
            'axes' in curr_metadata and
            'defaultMinimumConfidenceRating' in curr_metadata
    ):
        metadata['mean'] = curr_metadata['mean']
        metadata['scale'] = curr_metadata['scale']
        metadata['axes'] = curr_metadata['axes']
        metadata['defaultMinimumConfidenceRating'] = curr_metadata['defaultMinimumConfidenceRating']
        item_feature_axes = metadata['axes']

    recalculate_memory = True
    recalculate_performance = True

    # e.g. ONNX conversion failed
    if (file_int8 and not os.path.exists(file_int8)):
        file_int8 = None

    # For some model types (e.g. object detection) there is no keras model, so
    # we are unable to compute some of our stats with these methods
    if keras_model:
        # This describes the basic inputs and outputs, but skips over complex parts
        # such as transfer learning base models
        metadata['layers'] = describe_layers(keras_model)
        estimated_maccs = estimate_maccs_for_model(keras_model)
        # This describes the full model, so use it to determine if the architecture
        # has changed between runs
        if hasattr(keras_model, 'to_json'):
            metadata['kerasJSON'] = keras_model.to_json()
        # Only recalculate memory when model architecture has changed
        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
        ):
            recalculate_memory = False
        else:
            recalculate_memory = True

        if (
            curr_metadata and 'kerasJSON' in curr_metadata and 'metadataVersion' in curr_metadata
            and curr_metadata['metadataVersion'] == metadata['metadataVersion']
            and metadata['kerasJSON'] == curr_metadata['kerasJSON']
            and 'performance' in curr_metadata
            and curr_metadata['performance']
        ):
            metadata['performance'] = curr_metadata['performance']
            recalculate_performance = False
        else:
            recalculate_memory = True
            recalculate_performance = True
    else:
        metadata['layers'] = []
        estimated_maccs = -1
        # If there's no Keras model we can't tell if the architecture has changed, so recalculate memory every time
        recalculate_memory = True
        recalculate_performance = True

    if recalculate_performance:
        try:
            args = '/app/profiler/build/profiling '
            if file_float32:
                args = args + file_float32 + ' '
            if file_int8:
                args = args + file_int8 + ' '

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            if '{' in a and '}' in a:
                metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
                print('Calculating inferencing time OK', flush=True)
            else:
                print('Failed to calculate inferencing time:', a)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None

    float32_perf = None
    int8_perf = None

    if model_float32:
        try:
            if async_memory_profiling:
                print('Calculating float32 accuracy...', flush=True)
            else:
                print('Profiling float32 model...', flush=True)
            model_type = 'float32'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            float32_perf = profile_model(
                model_type=model_type,
                model=model_float32,
                model_file=file_float32,
                validation_dataset=validation_dataset,
                Y_test=Y_test,
                X_samples=X_samples,
                Y_samples=Y_samples,
                has_samples=has_samples,
                memory=memory,
                mode=mode,
                prepare_model_tflite_script=prepare_model_tflite_script,
                prepare_model_tflite_eon_script=prepare_model_tflite_eon_script,
                num_classes=len(class_names),
                train_dataset=train_dataset,
                Y_train=Y_train,
                test_dataset=test_dataset,
                Y_real_test=Y_real_test,
                akida_model_path=None,
                item_feature_axes=item_feature_axes,
                async_memory_profiling=async_memory_profiling,
                objdet_details=objdet_details)
            float32_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(float32_perf)
            metadata['modelIODetails'].append(get_io_details(model_float32, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite float32 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if model_int8:
        try:
            if async_memory_profiling:
                print('Calculating int8 accuracy...', flush=True)
            else:
                print('Profiling int8 model...', flush=True)
            model_type = 'int8'

            memory = None
            if not recalculate_memory:
                curr_metrics = list(filter(lambda x: x['type'] == model_type, curr_metadata['modelValidationMetrics']))
                if (len(curr_metrics) > 0):
                    memory = curr_metrics[0]['memory']

            int8_perf = profile_model(
                model_type=model_type,
                model=model_int8,
                model_file=file_int8,
                validation_dataset=validation_dataset,
                Y_test=Y_test,
                X_samples=X_samples,
                Y_samples=Y_samples,
                has_samples=has_samples,
                memory=memory,
                mode=mode,
                prepare_model_tflite_script=prepare_model_tflite_script,
                prepare_model_tflite_eon_script=prepare_model_tflite_eon_script,
                num_classes=len(class_names),
                train_dataset=train_dataset,
                Y_train=Y_train,
                test_dataset=test_dataset,
                Y_real_test=Y_real_test,
                akida_model_path=None,
                item_feature_axes=item_feature_axes,
                async_memory_profiling=async_memory_profiling,
                objdet_details=objdet_details)
            int8_perf['estimatedMACCs'] = estimated_maccs
            metadata['availableModelTypes'].append(model_type)
            metadata['modelValidationMetrics'].append(int8_perf)
            metadata['modelIODetails'].append(get_io_details(model_int8, model_type))
        except Exception as err:
            print('Unable to execute TensorFlow Lite int8 model:', flush=True)
            print(err, flush=True)
            traceback.print_exc()

    if file_akida:
        print('Profiling akida model...', flush=True)
        model_type = 'akida'

        program_size, total_nps, nodes = ei_tensorflow.brainchip.model.get_hardware_utilization(file_akida)
        flops, macs = ei_tensorflow.brainchip.model.get_macs_flops(keras_model)
        memory = {}
        memory['tflite'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        # only 'eon' is used, see comment in populateMetadataTemplate in
        # studio/client/project/pages/training-keras-ui.ts
        memory['eon'] = {
            'ram': -1,
            'rom': program_size,
            'arenaSize': 0,
            'modelSize': 0
        }
        if model_int8:
            io_details = get_io_details(model_int8, model_type)
        else:
            io_details = None
        akida_perf = profile_model(
            model_type=model_type,
            model=None,
            model_file=None,
            validation_dataset=validation_dataset,
            Y_test=Y_test,
            X_samples=X_samples,
            Y_samples=Y_samples,
            has_samples=has_samples,
            memory=memory,
            mode=mode,
            prepare_model_tflite_script=None,
            prepare_model_tflite_eon_script=None,
            num_classes=len(class_names),
            train_dataset=train_dataset,
            Y_train=Y_train,
            test_dataset=test_dataset,
            Y_real_test=Y_real_test,
            akida_model_path=file_akida,
            item_feature_axes=None,
            objdet_details=objdet_details,
            async_memory_profiling=False)
        sparsity = ei_tensorflow.brainchip.model.get_model_sparsity(file_akida, mode, validation_dataset,
                                                                    objdet_details=objdet_details)
        akida_perf['estimatedMACCs'] = macs
        metadata['availableModelTypes'].append(model_type)
        metadata['modelValidationMetrics'].append(akida_perf)
        # hack: let's grab input scailing from int8 model. Akida is expecting input to be only 8 bit
        # (or less - see Dense layer)
        if io_details is not None:
            metadata['modelIODetails'].append(io_details)

        # Also store against deviceSpecificPerformance
        metadata['deviceSpecificPerformance'] = {
            'brainchip-akd1000': {
                'model_quantized_int8_io.tflite': {
                    'latency': 0,
                    'ram': -1,
                    'rom': program_size,
                    'customMetrics': [
                        {
                            'name': 'nps',
                            'value': str(total_nps)
                        },
                        {
                            'name': 'sparsity',
                            'value': f'{sparsity:.2f}%'
                        },
                        {
                            'name': 'macs',
                            'value': str(int(macs))
                        }
                    ]
                }
            }
        }

    # Decide which model to recommend
    if file_akida:
        metadata['recommendedModelType'] = 'akida'
    else:
        recommended_model_type = get_recommended_model_type(float32_perf, int8_perf)
        metadata['recommendedModelType'] = recommended_model_type

    return metadata

def profile_tflite_file(file, model_type,
                        prepare_model_tflite_script,
                        prepare_model_tflite_eon_script,
                        calculate_inferencing_time,
                        calculate_is_supported_on_mcu,
                        calculate_non_cmsis):
    metadata = {
        'tfliteFileSizeBytes': os.path.getsize(file)
    }

    if calculate_inferencing_time:
        try:
            args = '/app/profiler/build/profiling ' + file

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
            print('Calculating inferencing time OK', flush=True)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None
    else:
        metadata['performance'] = None

    if calculate_is_supported_on_mcu:
        is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(file, log_messages=True)
        metadata['isSupportedOnMcu'] = is_supported_on_mcu
        metadata['mcuSupportError'] = mcu_support_error
    else:
        metadata['isSupportedOnMcu'] = True
        metadata['mcuSupportError'] = None

    if (metadata['isSupportedOnMcu']):
        metadata['memory'] = calculate_memory(file, model_type, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                                              calculate_non_cmsis=calculate_non_cmsis)
    return metadata

def check_if_model_runs_on_mcu(file, log_messages):
    is_supported_on_mcu = True
    mcu_support_error = None

    try:
        if log_messages:
            print('Determining whether this model runs on MCU...')

        # first we'll do a quick check against full TFLite. If the arena size is >6MB, we don't even pass it through
        # EON (fixes issues like https://github.com/edgeimpulse/edgeimpulse/issues/8838)
        full_tflite_result = subprocess.run(['/app/tflite-find-arena-size/find-arena-size', file], stdout=subprocess.PIPE)
        if (full_tflite_result.returncode == 0):
            stdout = full_tflite_result.stdout.decode('utf-8')
            msg = json.loads(stdout)

            arena_size = msg['arena_size']
            # more than 6MB
            if arena_size > 6 * 1024 * 1024:
                is_supported_on_mcu = False
                mcu_support_error = 'Calculated arena size is >6MB'

        # exit early
        if not is_supported_on_mcu:
            return is_supported_on_mcu, mcu_support_error

        result = subprocess.run(['/app/eon_compiler/compiler', '--verify', file], stdout=subprocess.PIPE)
        if (result.returncode == 0):
            stdout = result.stdout.decode('utf-8')
            msg = json.loads(stdout)

            arena_size = msg['arena_size']
            # more than 6MB
            if arena_size > 6 * 1024 * 1024:
                is_supported_on_mcu = False
                mcu_support_error = 'Calculated arena size is >6MB'
            else:
                is_supported_on_mcu = True
        else:
            is_supported_on_mcu = False
            stdout = result.stdout.decode('utf-8')
            if stdout != '':
                mcu_support_error = stdout
            else:
                mcu_support_error = 'Verifying model failed with code ' + str(result.returncode) + ' and no error message'
        if log_messages:
            print('Determining whether this model runs on MCU OK')
    except Exception as err:
        print('Determining whether this model runs on MCU failed:', flush=True)
        print(err, flush=True)
        traceback.print_exc()
        is_supported_on_mcu = False
        mcu_support_error = str(err)

    return is_supported_on_mcu, mcu_support_error
