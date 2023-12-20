import io, os
import tensorflow as tf
import numpy as np
import tflite
import sys

import ei_tensorflow.training
from ei_tensorflow.filter_outputs import output_redirector, print_filtered_output

def run_converter(converter: tf.lite.TFLiteConverter, redirect_streams=True):
    # The converter outputs some garbage that we don't want to end up in the user's log,
    # so we have to catch the c stdout/stderr and filter the things we don't want to keep.
    # TODO: Wrap this up more elegantly in a single 'with'

    # on macOS the functions below reference some kernel functions that can't be overriden
    # so just invoke the converter
    if (sys.platform == 'darwin' or not redirect_streams):
        return converter.convert()

    output_out = io.StringIO()
    output_err = io.StringIO()
    converted_model = None
    conversion_error = None
    with output_redirector('stdout', output_out), output_redirector('stderr', output_err):
        try:
            converted_model = converter.convert()
        except Exception as e:
            conversion_error = e
    print_filtered_output(output_out)
    print_filtered_output(output_err)
    if (conversion_error):
        raise conversion_error
    return converted_model

def convert_float32(concrete_func, keras_model, dir_path, filename):
    try:
        print('Converting TensorFlow Lite float32 model...', flush=True)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
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
        open(os.path.join(dir_path, filename), 'wb').write(tflite_model)
        return tflite_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite float32 model:')
        print(err)

# Declare a generator that can feed the TensorFlow Lite converter during quantization
def representative_dataset_generator(validation_dataset):
    def gen():
        for data, _ in validation_dataset.take(-1):
            yield [tf.convert_to_tensor([data])]
    return gen

def convert_int8_io_int8(concrete_func, keras_model, dataset_generator,
                         dir_path, filename, disable_per_channel = False):
    try:
        print('Converting TensorFlow Lite int8 quantized model...', flush=True)
        converter_quantize = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], keras_model)
        if disable_per_channel:
            converter_quantize._experimental_disable_per_channel = disable_per_channel
            print('Note: Per channel quantization has been automatically disabled for this model. '
                  'You can configure this in Keras (expert) mode.')
        converter_quantize.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_quantize.representative_dataset = dataset_generator
        # Force the input and output to be int8
        converter_quantize.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Restrict the supported types to avoid ops that are not TFLM compatible
        converter_quantize.target_spec.supported_types = [tf.dtypes.int8]
        converter_quantize.inference_input_type = tf.int8
        converter_quantize.inference_output_type = tf.int8
        tflite_quant_model = run_converter(converter_quantize)
        open(os.path.join(dir_path, filename), 'wb').write(tflite_quant_model)
        return tflite_quant_model
    except Exception as err:
        print('Unable to convert and save TensorFlow Lite int8 quantized model:')
        print(err)

def warn_about_issues(converted_model):
    """Explores the converted graph and warns about known issues.

    Args:
        converted_model: The model obtained from the TensorFlow Lite Converter
    """

    message_printed = False

    model = tflite.Model.GetRootAsModel(converted_model, 0)
    for s in range(0, model.SubgraphsLength()):
        graph = model.Subgraphs(s)
        for i in range(0, graph.OperatorsLength()):
            op = graph.Operators(i)
            op_code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            op_name = None
            for key, value in vars(tflite.BuiltinOperator).items():
                if (value == op_code):
                    op_name = key

            # Disabled the lines below, as we use TF2.4 for the ops here, but are at TF2.7
            # (which has some new ops) for our own models, so this triggers on e.g. YOLOX
            # even though everything is fine
            # if op_name is None:
            #     print(f'WARN: Model graph contains unknown op with code "{op_code}"')

            if op_name == 'FULLY_CONNECTED':
                # Check the dimensions of the output
                output = op.Outputs(0)
                out_tensor = graph.Tensors(output)
                # Inference will fail or give wrong results when there are
                # more than 2 output dimensions.
                if out_tensor.ShapeLength() > 2 and not message_printed:
                    # Get some information about the inputs so we can recommend a solution.
                    # The inputs are as follows:
                    #   0: actual input
                    #   1: weights
                    #   2: biases
                    # We want to check the dimensions of the actual input.
                    input = op.Inputs(0)
                    in_tensor = graph.Tensors(input)
                    # Get the shapes of the input and output tensors
                    in_shape_list = []
                    for t in range(0, in_tensor.ShapeLength()):
                        shape = in_tensor.Shape(t)
                        in_shape_list.append(shape)
                    out_shape_list = []
                    for t in range(0, out_tensor.ShapeLength()):
                        shape = out_tensor.Shape(t)
                        out_shape_list.append(shape)
                    final_dim_out = out_shape_list[-1]
                    final_dim_in = in_shape_list[-1]
                    msg = (
                        'WARNING: After conversion, this model contains a fully connected (Dense) layer whose output '
                        'has >2 dimensions.\n'
                        '\n'
                        f'Layer neurons: {final_dim_out} (e.g. `Dense({final_dim_out})`)\n'
                        f'Layer input shape: ({", ".join([str(int) for int in in_shape_list])},)\n'
                        f'Layer output shape: ({", ".join([str(int) for int in out_shape_list])},)\n'
                        '\n'
                        'EON Compiler and TensorFlow Lite for Microcontrollers do not currently support this and will '
                        'either fail to run or will output incorrect values for this layer. We are working on a fix '
                        'for this issue.\n'
                        '\n'
                        'Workarounds:\n'
                        "1) In simple cases you can reshape the layer's input so that it has two or fewer dimensions. "
                        f'For example, in Keras you might add `model.add(Reshape((-1, {final_dim_in},)))` '
                        'before the Dense layer. This may not work when multiple Dense layers are stacked.\n'
                        '\n'
                        '2) Dense layers can be replaced with Conv1D layers with `kernel_size=1`. '
                        f'For example, `Dense({final_dim_out})` can be replaced with `Conv1D({final_dim_out}, 1)`. '
                        'The output and model size will be identical.\n'
                        '\n'
                        'Please contact hello@edgeimpulse.com if you need help adapting your model.')
                    print()
                    print('================')
                    print(msg)
                    print('================')
                    print()
                    message_printed = True

def convert_to_tf_lite(model, best_model_path, dir_path, saved_model_dir, h5_model_path,
                      validation_dataset, model_input_shape, model_filenames_float,
                      model_filenames_quantised_int8, disable_per_channel = False, syntiant_target=False,
                      akida_model=False):
    model = ei_tensorflow.training.save_model(model, best_model_path, dir_path,
                                              saved_model_dir, h5_model_path, syntiant_target,
                                              akida_model=akida_model)
    dataset_generator = representative_dataset_generator(validation_dataset)
    concrete_func = ei_tensorflow.training.get_concrete_function(model, model_input_shape)

    tflite_model = convert_float32(concrete_func, model, dir_path, model_filenames_float)
    # Only need to call this on one model
    warn_about_issues(tflite_model)
    tflite_quant_model = convert_int8_io_int8(concrete_func, model, dataset_generator,
                                              dir_path, model_filenames_quantised_int8,
                                              disable_per_channel)

    return model, tflite_model, tflite_quant_model

def convert_jax_to_tflite_float32(jax_function, input_shape, redirect_streams=True):
    """
    Converts a JAX function into a tflite model and returns it.
    """
    input_shape = (1, *input_shape)
    converter = tf.lite.TFLiteConverter.experimental_from_jax(
        [jax_function],
        [[('', np.zeros(input_shape))]])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter.target_spec.supported_types = [
        tf.dtypes.float32,
        tf.dtypes.int8
    ]
    tflite_model = run_converter(converter, redirect_streams)
    return tflite_model

def convert_jax_to_tflite_int8(jax_function, input_shape, dataset_generator, redirect_streams=True):
    """
    Converts a JAX function into a tflite model and returns it.
    """
    input_shape = (1, *input_shape)
    converter = tf.lite.TFLiteConverter.experimental_from_jax(
        [jax_function],
        [[('', np.zeros(input_shape))]])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_generator
    # Force the input and output to be int8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Restrict the supported types to avoid ops that are not TFLM compatible
    converter.target_spec.supported_types = [tf.dtypes.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = run_converter(converter, redirect_streams)
    return tflite_model
