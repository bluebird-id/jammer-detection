import io, os
import tensorflow as tf
import numpy as np
import onnx
import json
import traceback
from .onnx_input_order_convertor import order_conversion
from onnx_tf.backend import prepare # https://github.com/onnx/onnx-tensorflow
from ..conversion import warn_about_issues, representative_dataset_generator, run_converter

def convert_int8_io_int8(tf_model_path, dataset_generator,
                                         tflite_file, disable_per_channel = False):
    converter_quantize = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    if disable_per_channel:
        converter_quantize._experimental_disable_per_channel = disable_per_channel
        print('    Note: Per channel quantization has been automatically disabled for this model.')
    converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quantize.representative_dataset = dataset_generator
    # Force the input and output to be int8
    converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
    converter_quantize.inference_input_type = tf.int8
    converter_quantize.inference_output_type = tf.int8
    tflite_quant_model = run_converter(converter_quantize)
    with open(tflite_file, 'wb') as f:
        f.write(tflite_quant_model)
    return tflite_quant_model


def convert_float32(tf_model_path, tflite_file):
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter.target_spec.supported_types = [
        tf.dtypes.float32,
        tf.dtypes.int8
    ]
    tflite_model = run_converter(converter)
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    return tflite_model

# Convert an ONNX file into TFLite files
# if you pass in None for tflite_file_int8 it'll be skipped
# This also auto-converts NCHW to NHWC (using a bunch of Transpose layers though, so not very efficient)
def onnx_to_tflite(onnx_file, tflite_file, tflite_file_int8, validation_dataset, tf_model_path='/tmp/savedmodel'):
    if tflite_file_int8 and not validation_dataset:
        raise Exception('onnx_to_tflite should be called with a validation_dataset if tflite_file_int8 is provided')

    tmp_onnx_file = '/tmp/model.onnx'

    # 1. add shape info (and save)
    try:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_file)), tmp_onnx_file)
    except Exception as e:
        tmp_onnx_file = onnx_file
        msg = traceback.format_exc(limit=1).split('\n')
        msg = [i for i in msg if i != '']
        print('WARN: Failed to run infer_shapes', msg[-1])
        print('')

    # 2. load the graph to see if we need to do NHWC conversion
    onnx_graph = onnx.load(tmp_onnx_file)

    input_all = [node.name for node in onnx_graph.graph.input]
    input_initializer =  [node.name for node in onnx_graph.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))

    if (len(net_feed_input) != 1):
        raise Exception('WARN: Graph has ' + str(len(net_feed_input)) + ' inputs, only one supported (inputs found: ' + ','.join(net_feed_input) + ')')

    input_tensor_name = net_feed_input[0]
    input_tensor = None
    for input in onnx_graph.graph.input:
        if input.name == input_tensor_name:
            input_tensor = input
            break

    if input_tensor == None:
        raise Exception('WARN: Could not find input tensor with name "' + input_tensor_name + '"')

    batch_dim = list(input_tensor.type.tensor_type.shape.dim)[0]
    if batch_dim.dim_value == 0 and batch_dim.dim_param != '':
        print('INFO: Input tensor does not have a fixed batch dimension, but is named "' + batch_dim.dim_param + '", setting to "1"')
        batch_dim.dim_value = 1

    input_shape = [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
    if (input_shape[0] != 1):
        raise Exception('Expected an input shape with batch size 1, but got: ' + json.dumps(input_shape) + '. ' +
                            'If you have symbolic dimensions or dynamic shapes in your network, see ' +
                            'https://onnxruntime.ai/docs/tutorials/mobile/helpers/make-dynamic-shape-fixed.html#making-a-symbolic-dimension-fixed to make these fixed first.')

    if (len(input_shape) == 4 and input_shape[0] == 1 and (input_shape[1] == 1 or input_shape[1] == 3)):
        input_op_names_and_order_dims = {}
        input_op_names_and_order_dims[input_tensor.name] = [ 0, 2, 3, 1 ]

        # looks like NCHW
        onnx_graph = order_conversion(
            onnx_graph=onnx_graph,
            input_op_names_and_order_dims=input_op_names_and_order_dims,
            non_verbose=True
        )
        input_shape = [d.dim_value for d in onnx_graph.graph.input[0].type.tensor_type.shape.dim]

    # remove the batch
    input_shape = tuple(input_shape[1:])

    onnx.checker.check_model(onnx_graph)

    # this is set to 3, but requires 100x the time to export to graph in that case,
    # so temporary set it to 0
    tf.autograph.set_verbosity(0)

    tf_rep = prepare(onnx_graph, device='cpu')
    tf_rep.export_graph(tf_model_path)

    # and set it back
    tf.autograph.set_verbosity(3)

    print('Converting to TensorFlow Lite (float32)...')
    tflite_model = convert_float32(tf_model_path, tflite_file)
    # Only need to call this on one model
    warn_about_issues(tflite_model)
    print('Converting to TensorFlow Lite (float32) OK')

    if tflite_file_int8:
        try:
            dataset_generator = representative_dataset_generator(validation_dataset)

            print('Converting to TensorFlow Lite (quantized)...')
            convert_int8_io_int8(tf_model_path, dataset_generator,
                tflite_file_int8,
                disable_per_channel=False)
            print('Converting to TensorFlow Lite (quantized) OK')
            print('')
        except Exception as e:
            if (os.path.exists(tflite_file_int8)):
                os.unlink(tflite_file_int8)
            print('WARN: Failed to convert to TensorFlow Lite (quantized):', e)
