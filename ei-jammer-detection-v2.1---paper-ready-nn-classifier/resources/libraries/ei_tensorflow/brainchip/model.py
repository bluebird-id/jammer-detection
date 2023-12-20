import os, shutil, time
import tensorflow as tf
import numpy as np
import akida
from cnn2snn import convert

from typing import Optional
from ei_shared.types import ObjectDetectionDetails

def get_akida_converted_model(model, input_shape):
    # https://doc.brainchipinc.com/api_reference/cnn2snn_apis.html#convert
    # The input_scaling param works like this:
    # input_akida = input_scaling[0] * input_keras + input_scaling[1]
    # It needs to be matched by a similar conversion when we perform inference.
    input_is_image = False
    input_scaling = None
    if len(input_shape) == 3:
        input_is_image = True
        # Don't set input scaling explicitly if there is a rescaling layer present
        # TODO: Make an explicit function for doing these layer checks
        rescaling = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling)]
        if len(rescaling) == 0:
            input_scaling = (255, 0)

    return convert(model, input_is_image=input_is_image, input_scaling=input_scaling)

def convert_akida_model(dir_path, model, model_path, input_shape):
    if not isinstance(model, (akida.Model)):
        print('Converting to Akida model...')
        print('')
        model_akida = get_akida_converted_model(model, input_shape)
        print('Converting to Akida model OK')
        print('')
    else:
        print('Requested model is already converted to Akida')
        model_akida = model

    # try to map the model on the virtual AKD1000 device
    model_akida.map(akida.AKD1000())

    model_akida.summary()

    print('Saving Akida model...')
    model_akida.save(os.path.join(dir_path, model_path))
    print('Saving Akida model OK...')

def save_akida_model(akida_model, path):
    print('Saving Akida model...', flush=True)
    to_save = tf.keras.models.clone_model(akida_model)
    to_save.save(path)
    print('Saving Akida model OK', flush=True)

def load_akida_model(path):
    return akida.Model(path)

def make_predictions(mode, model_path, validation_dataset,
                    Y_test, train_dataset, Y_train, test_dataset, Y_real_test, num_classes, output_directory,
                    objdet_details: Optional[ObjectDetectionDetails]=None):
    prediction_train = None
    prediction_test = None

    if mode == 'classification':
        prediction = predict(model_path, validation_dataset, len(Y_test))
        if (train_dataset is not None) and (Y_train is not None):
            prediction_train = predict(model_path, train_dataset, len(Y_train))
        if (test_dataset is not None) and (Y_real_test is not None):
            prediction_test = predict(model_path, test_dataset, len(Y_real_test))
    elif mode == 'object-detection':
        assert objdet_details is not None, "objdet_details must be provided for object detection mode"
        if objdet_details.last_layer == 'fomo':
            prediction = predict_segmentation(model_path, validation_dataset, len(Y_test))
        elif objdet_details.last_layer == 'yolov2-akida':
            prediction = akida_predict_yolov2(model_path, validation_dataset, Y_test, len(Y_test), num_classes, output_directory)
        else:
            raise ValueError('Unsupported object detection mode: ' + objdet_details.last_layer)
    else:
        raise ValueError('Unsupported mode for profiling: ' + mode)

    return prediction, prediction_train, prediction_test

def process_input(data, input_is_4bits = False):
    # we are doing `clip` below, because if data is negative,
    # conversion to uint8 will make it 255
    if input_is_4bits:
        # Akida approach to quantization
        # recommended by BrainChip team
        data = data * 15
        data = np.clip(data, 0, 255)
        data = np.uint8(data / 16)
    else:
        data = data * 255
        data = np.clip(data, 0, 255)
        data = np.uint8(data)

    return data

def akida_predict_yolov2(model_path, validation_dataset, Y_test, dataset_length, num_classes, output_directory):
    import pickle

    model = load_akida_model(model_path)
    input_shape = model.input_shape
    width = input_shape[0]
    height = input_shape[1]
    with open(os.path.join(output_directory, "akida_yolov2_anchors.pkl"), 'rb') as handle:
        anchors = pickle.load(handle)

    last_log = time.time()

    pred_y = []
    for batch, _ in validation_dataset.take(-1):
        for item in batch:
            item = (item * 255)
            item = np.array(item)
            output = model.predict(item.astype('uint8'))[0]
            h, w, c = output.shape
            output = output.reshape((h, w, len(anchors), 4 + 1 + num_classes))
            rect_label_scores = process_output_yolov2(output, (width, height), 2, anchors)
            pred_y.append(rect_label_scores)
            current_time = time.time()
            if last_log + 10 < current_time:
                print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
                last_log = current_time

    result = np.array(pred_y)
    return result

def predict_segmentation(model_path, validation_dataset, dataset_length):
    """Runs an Akida model across a set of inputs"""
    model = load_akida_model(model_path)

    last_log = time.time()

    pred_y = []
    for item, _ in validation_dataset.take(-1):
        item = (item * 255)
        item = np.expand_dims(item, axis=0)
        output = model.predict(item.astype('uint8'))
        output = np.squeeze(output)
        pred_y.append(output)
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def predict(model_path, validation_dataset, dataset_length):
    """Runs an Akida model across a set of inputs"""
    model = load_akida_model(model_path)

    last_log = time.time()
    input_is_4bit = model.layers[0].input_bits == 4

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item = process_input(item, input_is_4bit)
        item = np.expand_dims(item, axis=0)
        output = model.predict(item)
        output = np.squeeze(output)
        pred_y.append(output)
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def get_model_sparsity(model_file, mode, dataset,
                       objdet_details: Optional[ObjectDetectionDetails]=None):
    """Returns a sparsity of the model, using dataset as an input.

    Returned sparsity is in percents.
    Please refer to:
    https://doc.brainchipinc.com/api_reference/akida_apis.html#sparsity
    """
    model = load_akida_model(model_file)

    input = []
    if mode == 'classification':
        for item, _ in dataset.take(-1).as_numpy_iterator():
            item = process_input(item, model.layers[0].input_bits == 4)
            input.append(item)
    elif mode == 'object-detection':
        assert objdet_details is not None, "objdet_details must be provided for object detection mode"
        if objdet_details.last_layer == 'fomo':
            for item, _ in dataset.take(-1):
                item = process_input(item, model.layers[0].input_bits == 4)
                input.append(item)

    try:
        raw_sparsity = akida.evaluate_sparsity(model, np.array(input, np.uint8))
    except Exception as err:
        print("EI_LOG_LEVEL=error ERROR: Can't calculate model sparsity: " + str(err))
        return 0

    # calculate average sparsity for model
    num = 0
    sparsity = 0
    for s in raw_sparsity.values():
        if s is not None:
            sparsity += s
            num += 1
    sparsity /= num
    # return sparsity in %
    sparsity *= 100

    return sparsity

# This function is backported from akida_models 1.1.9 Python package
def get_macs_flops(model: tf.keras.Model, verbose=False):
    """
    Calculate FLOPS and MACs for a tf.keras.Model or tf.keras.Sequential model
    in inference mode.

    It uses tf.compat.v1.profiler under the hood.

    Args:
        model (:obj:`keras.Model`): the model to evaluate
        verbose (bool): display MACS for each operation

    Returns:
        :obj:`tf.compat.v1.profiler.GraphNodeProto`: object containing the FLOPS
    """

    from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph)

    # Prepare a constant input to pass to the profiler
    input_shape = model.inputs[0].shape.as_list()
    input_shape[0] = 1
    x = tf.constant(tf.fill(input_shape, 1))
    if not isinstance(model, (tf.keras.models.Sequential, tf.keras.models.Model)):
        raise ValueError("Calculating FLOPS is only supported for `tf.keras.Model`"
                         "and `tf.keras.Sequential` instances.")

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(x)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops_obj = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    tf.compat.v1.reset_default_graph()

    try:
        flops = flops_obj.total_float_ops
    except Exception as err:
        print(f"EI_LOG_LEVEL=warn WARNING: Can't calculate FLOPS: {err}")
        flops = 0

    if verbose:
        def display_children_macs(nodes):
            for node in nodes:
                print(f"{node.name}: {node.total_float_ops / 2:e} MACS")
                display_children_macs(node.children)
        # Recursively display MACS by node (i.e. operation)
        display_children_macs(flops_obj.children)

    # We divide FLOPS by 2 to obtain an estimate of Multiply and Accumulate (MACS)
    macs = flops_obj.total_float_ops / 2

    return flops, macs


def get_hardware_utilization(model_file):
    """Returns utilization of AKD1000 NSoC.

    Returned value is a tuple of: program_size, nps, nodes
    Program size is exact size of the model (not the FBZ file) that need to be stored
    NPs is a number of Neural Processors
    Nodes is a number of used nodes (each node consist of 4 NPs)
    """

    try:
        model = akida.Model(model_file)
    except Exception as err:
        print("EI_LOG_LEVEL=error ERROR: Can't map model to AKD1000 NSoC! Can't calculate program size, number of NPs and nodes!")
        print("EI_LOG_LEVEL=error ERROR: " + str(err))
        # report -1 (Flash size AKA program size = N/A), and 0 nodes and NPs
        return -1, 0, 0

    try:
        model.map(akida.AKD1000(), hw_only=True)
    except Exception as err:
        print("EI_LOG_LEVEL=warn WARNING: Requested model can't be fully mapped to hardware. Reason:")
        print("EI_LOG_LEVEL=warn WARNING: " + str(err))
        print("EI_LOG_LEVEL=warn WARNING: Reported program size, number of NPs and nodes may not be accurate!")

    # after mapping model onto the hardware (in our case the virtual AKD1000 NSoC)
    # we can iterate over all sequences to collect their sizes and number of used NPs.
    # Some models couldn't be mapped fully on hardware. In such cases, there will be a sewuences
    # (one or a few model layers) that will be processed on the host CPU, so we can't count them.
    total_nps = 0
    program_size = 0
    # iterate through mapped sequences
    for i, seq in enumerate(model.sequences, start=1):
        # if the sequence is not mapped to hardware, then skip it
        if seq.backend != akida.BackendType.Hardware:
            continue
        # get program size of the current sequence
        program_size += len(seq.program)
        # iterate through passes in sequence
        # see pass info: https://doc.brainchipinc.com/api_reference/akida_apis.html?highlight=passes#akida.Pass
        for j, seq_pass in enumerate(seq.passes, start=1):
            total_nps_pass = 0
            for n, layer in enumerate(seq_pass.layers, start=1):
                try:
                    nps = [conf.ident for conf in layer.mapping.nps]
                except:
                    continue
                if i==1 and j==1 and n==1:
                    # first layer NPs is in the HRC and use default NPs
                    continue
                total_nps_pass += len(nps)
            total_nps += total_nps_pass

    # one node consist of 4 NPs, so if we are using 9 NPs (2,25 node) we need to report 3 nodes
    # Node is a unit used for licensing BrainChip's IP, so it's important to see how many nodes
    # you need to license as a customer
    nodes = int(total_nps / 4)
    if total_nps % 4:
        nodes += 1

    return (program_size, total_nps, nodes)

def convert_bbox_to_anchors(Y, image_width, image_height):
    anchors = []

    for ix in range(0, len(Y)):
        # Create one data dictionary for each image
        data = {"boxes": []}
        labels = Y[ix]['boundingBoxes']

        # TODO: last value (3) is image_channels. Seems to be not used in the processing of YOLOv2 output
        data['image_shape'] = (image_width, image_height, 3)
        labels_text = []
        # All the labels found in one image
        for l in labels:
            # Dimensions of one bounding box:
            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            # Class x_center y_center width height
            x_center = (x + (w / 2)) / image_width
            y_center = (y + (h / 2)) / image_height
            width = w / image_width
            height = h / image_height

            # Dimensions of one bounding box as required by Brainchip
            x1 = x
            x2 = x1 + w
            y1 = y
            y2 = y1 + h

            # Create another dictionary to hold bounding box data and labels
            box = {}

            box['label'] = str(l['label'] - 1)
            box['x1'] = int(round(float(x1)))
            box['y1'] = int(round(float(y1)))
            box['x2'] = int(round(float(x2)))
            box['y2'] = int(round(float(y2)))

            # After the above 5 pieces of info are added to the key, append to "box"
            # starts again if more than one bounding box + label in one image
            # TODO: is len(box) different than 5 anytime?
            if len(box) == 5:
              data["boxes"].append(box)

        # If there are any bounding boxes located in the image, append to anchors, otherwise skip append
        if len(data["boxes"]) != 0:
            anchors.append(data)

    return(anchors)

def process_output_yolov2(output_data, img_shape, num_classes, anchors, minimum_confidence_rating=None):
    from akida_models.detection.processing import decode_output
    from ei_tensorflow.inference import object_detection_nms
    # OUTPUT example: type: float32[1,7,7,5,7]
    # output: (N, grid_height, grid_width, anchors_box, 4 + 1 + num_classes)
    # The model output can be reshaped to a more natural shape of:
    # (grid_height, grid_width, anchors_box, 4 + 1 + num_classes)
    # where the “4 + 1” term represents the coordinates of the estimated bounding boxes
    # (top left x, top left y, width and height) and a confidence score. In other words,
    # the output channels are actually grouped by anchor boxes, and in each group one channel provides
    # either a coordinate, a global confidence score or a class confidence score.
    # This process is done automatically in the decode_output function.

    raw_width = img_shape[0]
    raw_height = img_shape[1]
    max_box_per_image = 10
    pred_boxes = decode_output(output_data, anchors, num_classes)

    # get lists of scores, labels and boxes
    score = np.array([box.get_score() for box in pred_boxes])
    pred_labels = np.array([box.get_label() for box in pred_boxes])
    if len(pred_boxes) > 0:
        # print(f"PRED BOX    ::: {pred_boxes}")
        pred_boxes = np.array([[
            box.x1 * raw_width, box.y1 * raw_height, box.x2 * raw_width,
            box.y2 * raw_height
        ] for box in pred_boxes])
    else:
        pred_boxes = np.array([[]])

    # sort the boxes and the labels according to scores
    score_sort = np.argsort(-score)
    pred_labels = pred_labels[score_sort]
    pred_boxes = pred_boxes[score_sort]

    raw_scores = list(zip(pred_boxes, pred_labels, score))
    nms_scores = object_detection_nms(raw_scores, raw_width, 0.4)

    return nms_scores
