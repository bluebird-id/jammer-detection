import tensorflow as tf
import numpy as np
import os, json, time, threading, shutil, zipfile
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy.lib.format as fmt
from collections import Counter
import math
from tensorflow.keras.callbacks import Callback
from typing import Optional

import ei_tensorflow.utils
from ei_shared.types import ObjectDetectionLastLayer
import ei_tensorflow.gpu
from ei_augmentation.object_detection import Augmentation

# Loads a features file, mmap's if size is above 128MiB
def np_load_file_auto_mmap(file):
    # 128MiB seems like a safe number to always load fully into memory
    # e.g. small feature sets, labels, etc. we can probably up it if we actually calculate the used memory
    if (os.path.getsize(file) > 128 * 1024 * 1024):
        return np.load(file, mmap_mode='r')
    else:
        return np.load(file)

# Since the data files are too large to load into memory, we must split and shuffle them by writing their contents to
# new files in random order. The new files can be memory mapped in turn.
def split_and_shuffle_data(y_type, classes, classes_values, mode, seed, dir_path, test_size=0.2,
                           X_train_features_path='X_train_features.npy', y_train_path='y_train.npy',
                           X_train_raw_path=None, stratify_sample=False,
                           output_directory='/tmp',
                           model_input_shape=None,
                           custom_validation_split=False,
                           custom_validation_split_path='custom_validation_split.json'):
    # This is where the split data will be written
    X_train_output_path = os.path.join(output_directory, 'X_split_train.npy')
    X_train_raw_output_path = os.path.join(output_directory, 'X_split_train_raw.npy')
    X_test_output_path = os.path.join(output_directory, 'X_split_test.npy')
    Y_train_output_path = os.path.join(output_directory, 'Y_split_train.npy')
    Y_test_output_path = os.path.join(output_directory, 'Y_split_test.npy')

    X = None
    X_raw = None
    Y = None

    # assume we are not stratifying labels by default.
    split_stratify_labels = None

    # Load the training data
    if y_type == 'structured':
        if X_train_raw_path:
            raise Exception('Raw input is not yet supported for structured Y data')

        X = np_load_file_auto_mmap(os.path.join(dir_path, 'X_train_features.npy'))
        Y = ei_tensorflow.utils.load_y_structured(dir_path, 'y_train.npy', len(X))
        if custom_validation_split:
            Y_sample_ids = []
            for row in Y:
                Y_sample_ids.append(row['sampleId'])

    elif y_type == 'npy':
        X = np_load_file_auto_mmap(os.path.join(dir_path, X_train_features_path))
        Y_file = np_load_file_auto_mmap(os.path.join(dir_path, y_train_path))
        Y = Y_file[:,0]
        Y_sample_ids = Y_file[:,1]

        if X_train_raw_path:
            X_raw = np_load_file_auto_mmap(os.path.join(dir_path, X_train_raw_path))

        # If we are building the sample with stratification; i.e, strictly enforcing that
        # the label distribution matches between train/test; we need to record the raw Y
        # values now before they are mutated.
        if stratify_sample:
            split_stratify_labels = Y

        # Do this before writing new splits to disk so that the resulting mmapped array has categorical values,
        # otherwise it will be difficult to generate performance stats, which depend on numpy arrays.
        if mode == 'classification':
            Y = tf.keras.utils.to_categorical(Y - 1, num_classes=classes)
        elif mode == 'regression':
            # for regression we want to map to the real values
            Y = np.array([ float(classes_values[y - 1]) for y in Y ])
            if np.isnan(Y).any():
                print('Your dataset contains non-numeric labels. Cannot train regression model.')
                exit(1)
    else:
        print('Invalid value for y_type')
        exit(1)

    if (model_input_shape):
        X = X.reshape(tuple([ X.shape[0] ]) + model_input_shape)

    # Facilitates custom split
    X_train_ids_assigned = []
    X_test_ids_assigned = []
    X_ids_unassigned = []

    if custom_validation_split:
        validation_split_metadata = ei_tensorflow.utils.load_validation_split_metadata(dir_path, custom_validation_split_path)
        if validation_split_metadata is not None:
            print('Using custom validation split...')
            # Split X and Y into assigned categories
            for ix, sample_id in enumerate(Y_sample_ids):
                sample_id = str(sample_id)
                if not sample_id in validation_split_metadata:
                    X_ids_unassigned.append(ix)
                    continue

                if validation_split_metadata[sample_id] == 'train':
                    X_train_ids_assigned.append(ix)
                elif validation_split_metadata[sample_id] == 'validation':
                    X_test_ids_assigned.append(ix)
                else:
                    X_ids_unassigned.append(ix)
        else:
            custom_validation_split = False

    # No custom validation split? Perform normal train/test split on all samples
    if not custom_validation_split:
        X_ids_unassigned = list(range(len(X)))

    # Perform train/test split on any samples not explicitly assigned a category
    if len(X_ids_unassigned) > 0:
        X_train_ids, X_test_ids, Y_train_ids, Y_test_ids = train_test_split(X_ids_unassigned,
                                                                            X_ids_unassigned.copy(),
                                                                            test_size=test_size,
                                                                            random_state=seed,
                                                                            stratify=split_stratify_labels)
    else:
        X_train_ids = []
        X_test_ids = []

    # Merge groups
    X_train_ids = X_train_ids + X_train_ids_assigned
    X_test_ids = X_test_ids + X_test_ids_assigned

    # If we had a custom validation split, shuffle again
    if custom_validation_split:
        X_train_ids = shuffle(X_train_ids, random_state=seed)
        Y_train_ids = X_train_ids.copy()
        X_test_ids = shuffle(X_test_ids, random_state=seed)
        Y_test_ids = X_test_ids.copy()

    # Generates a header for the .npy file
    def get_header(array, new_length):
        new_shape = (new_length,) + array.shape[1:]
        return {'descr': fmt.dtype_to_descr(array.dtype), 'fortran_order': False, 'shape': new_shape}

    # Saves a subset of an array's indexes to a numpy file, the subset and order specified by an array of ints
    def save_to_npy(array, indexes, file_path):
        header = get_header(array, len(indexes))
        with open(file_path, 'wb') as f:
            fmt.write_array_header_2_0(f, header)
            for ix in indexes:
                f.write(array[ix].tobytes('C'))

    save_to_npy(X, X_train_ids, X_train_output_path)
    save_to_npy(X, X_test_ids, X_test_output_path)

    if X_train_raw_path:
        # We only need the train split for the raw data, since test will not have augmentations applied
        save_to_npy(X_raw, X_train_ids, X_train_raw_output_path)

    if y_type == 'structured':
        # The structured data is just handled in memory.
        # Load these from JSON and then split Y_structured_train using the same method as above.
        Y_train = [Y[i] for i in Y_train_ids]
        Y_test = [Y[i] for i in Y_test_ids]

        with open(Y_train_output_path, 'w') as f:
            f.write(json.dumps(Y_train))
        with open(Y_test_output_path, 'w') as f:
            f.write(json.dumps(Y_test))

    elif y_type == 'npy':
        save_to_npy(Y, Y_train_ids, Y_train_output_path)
        save_to_npy(Y, Y_test_ids, Y_test_output_path)

    return load_split_and_shuffled_data(output_directory, y_type, X_train_raw_path=X_train_raw_path)

def load_split_and_shuffled_data(data_directory, y_type, X_train_raw_path=None):
    # This is where the split data will be written
    X_train_output_path = os.path.join(data_directory, 'X_split_train.npy')
    X_train_raw_output_path = os.path.join(data_directory, 'X_split_train_raw.npy')
    X_test_output_path = os.path.join(data_directory, 'X_split_test.npy')
    Y_train_output_path = os.path.join(data_directory, 'Y_split_train.npy')
    Y_test_output_path = os.path.join(data_directory, 'Y_split_test.npy')

    if X_train_raw_path:
        # We only need the train split for the raw data, since test will not have augmentations applied
        X_train_raw = np_load_file_auto_mmap(X_train_raw_output_path)
    else:
        X_train_raw = None

    X_train = np_load_file_auto_mmap(X_train_output_path)
    X_test = np_load_file_auto_mmap(X_test_output_path)
    Y_train = None
    Y_test = None

    if  y_type == 'structured':
        with open(Y_train_output_path, 'r') as f:
            Y_train = json.loads(f.read())
        with open(Y_test_output_path, 'r') as f:
            Y_test = json.loads(f.read())
    elif y_type == 'npy':
        Y_train = np_load_file_auto_mmap(Y_train_output_path)
        Y_test = np_load_file_auto_mmap(Y_test_output_path)

    return X_train, X_test, Y_train, Y_test, X_train_raw

# Take a folder with data (.npy files) and turn it into a dataset. If the data folder contains 'X_split_train.npy' then
# we'll assume the data is already split before, and we load data as-is. Otherwise we first split and shuffle the data.
# If flatten_dataset is True we'll transform the X shape to be a 2D vector. E.g. if you have a shape (10, 96, 96, 1) we'll
# reshape to (10, 9216). This is required for f.e. FOMO right now.
def get_dataset_from_folder(input,
                            data_directory,
                            RANDOM_SEED,
                            online_dsp_config,
                            input_shape,
                            ensure_determinism=False):
    y_type = input.yType
    raw_data_path = 'X_train_raw.npy' if hasattr(input, 'onlineDspConfig') else None
    classes_values = input.classes
    classes = 1 if input.mode == 'regression' else len(classes_values)
    mode = input.mode
    custom_validation_split = input.customValidationSplit if hasattr(input, 'customValidationSplit') else False

    # if we have split data already in the out directory, then use that
    if (os.path.exists(os.path.join(data_directory, 'X_split_train.npy'))):
        X_train, X_test, Y_train, Y_test, X_train_raw = load_split_and_shuffled_data(data_directory, y_type, X_train_raw_path=raw_data_path)

        # A subset of training data used for the feature explorer (we *load* it here, don't copy)
        has_samples, X_samples, Y_samples = load_samples(data_directory)
    else:
        # otherwise we'll split it ourselves
        print('Splitting data into training and validation sets...', flush=True)
        X_train, X_test, Y_train, Y_test, X_train_raw = split_and_shuffle_data(
            y_type, classes, classes_values, mode, RANDOM_SEED, data_directory,
            test_size=input.trainTestSplit,
            X_train_raw_path=raw_data_path,
            stratify_sample=False,
            model_input_shape=input_shape,
            custom_validation_split=custom_validation_split)
        print('Splitting data into training and validation sets OK', flush=True)

        # A subset of training data used for the feature explorer (we copy to /tmp here)
        has_samples, X_samples, Y_samples = get_samples(data_directory)

    # reshape X_samples into proper shape
    if has_samples:
        if (len(X_samples.shape) == 2 and np.prod(input_shape) == X_samples.shape[1]):
            X_samples = X_samples.reshape((X_samples.shape[0], ) + tuple(input_shape))

    if (input.flattenDataset):
        X_train = X_train.reshape((X_train.shape[0], int(X_train.size / X_train.shape[0])))
        X_test = X_test.reshape((X_test.shape[0], int(X_test.size / X_test.shape[0])))

    object_detection_last_layer = input.objectDetectionLastLayer if input.mode == 'object-detection' else None
    obj_detection_augmentation = input.objectDetectionAugmentation
    object_detection_batch_size = None

    # Get batch size for SSD models.
    # "Standard" models (i.e. non object-detection + FOMO) pass this via expert mode.
    if input.mode == 'object-detection' and object_detection_last_layer != 'fomo':
        object_detection_batch_size = input.objectDetectionBatchSize

    train_dataset, validation_dataset, samples_dataset = get_datasets(X_train, Y_train, X_test, Y_test,
                        has_samples, X_samples, Y_samples, mode, classes,
                        input_shape, X_train_raw, online_dsp_config,
                        obj_detection_augmentation,
                        object_detection_last_layer,
                        object_detection_batch_size=object_detection_batch_size,
                        ensure_determinism=ensure_determinism)

    return train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples

# Feeds values from our memory mapped training data into the tensorflow dataset
def create_generator_standard(X_values, Y_values):
    data_length = len(X_values)
    def gen():
        for ix in range(data_length):
            yield X_values[ix], Y_values[ix]
    return gen

def get_dataset_standard(X_values, Y_values):
    memory_used = X_values.size * X_values.itemsize

    # if we have <1GB of data then just load into memory
    # the train jobs have at least 8GiB of RAM, so this should be fine
    if (memory_used < 1 * 1024 * 1024 * 1024):
        return tf.data.Dataset.from_tensor_slices((X_values, Y_values))
    # otherwise we'll page the data in using a generator,
    # will be revisited in https://github.com/edgeimpulse/edgeimpulse/issues/3847 to lower memory reqs
    else:
        # Using the 'args' param of 'from_generator' results in a memory leak, so we instead use a function that
        # returns a generator that wraps the data arrays.
        return tf.data.Dataset.from_generator(create_generator_standard(X_values, Y_values),
                                                output_types=(tf.float32, tf.float32),
                                                output_shapes=(tf.TensorShape(X_values[0].shape),
                                                            tf.TensorShape(Y_values[0].shape)))

# Feeds values from our memory mapped training data into the tensorflow dataset
def create_generator_object_detection(X_values: np.memmap, width: int, height: int, num_channels: int,
                                      Y_values: list, num_classes: int, augment: bool):
    data_length = len(X_values)
    if augment:
        augmenter = Augmentation(width, height, num_channels)
    def gen():
        for ix in range(data_length):
            x, raw_boxes = X_values[ix], Y_values[ix]['boundingBoxes']
            if augment:
                x, raw_boxes = augmenter.augment(x, raw_boxes)
            # Not sure why but if the values are not unpacked in this manner the data does not pass the
            # output_signature test in get_dataset_object_detection
            boxes, classes = ei_tensorflow.utils.process_bounding_boxes(
                raw_boxes, width, height, num_classes)
            yield x, (boxes, classes)
    return gen

def get_dataset_object_detection(X_values: np.memmap, width: int, height: int, num_channels: int, Y_values: list,
                                 num_classes: int, augment: bool):
    # Using the 'args' param of 'from_generator' results in a memory leak, so we instead use a function that
    # returns a generator that wraps the data arrays.
    return tf.data.Dataset.from_generator(
        create_generator_object_detection(X_values, width, height, num_channels, Y_values, num_classes, augment),
        output_signature=(
            tf.TensorSpec(shape=X_values[0].shape, dtype=tf.float32),
            (tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.RaggedTensorSpec(shape=(None, num_classes), dtype=tf.float32))))

def get_reshape_function(reshape_to):
    def reshape(image, label):
        return tf.reshape(image, reshape_to), label
    return reshape

def get_samples(dir_path):
    # DSP blocks generate a set of samples (max. 2000) to display in the feature explorer.
    # Load these samples, then when profiling make a prediction for each sample to show how the sample is classified in
    # the feature explorer.
    X_samples_path = os.path.join(dir_path, 'X_train_samples.npy')
    X_samples = None
    Y_samples_path = os.path.join(dir_path, 'y_samples.npy')
    Y_samples = None
    has_samples = False
    try:
        x_tmp_file = os.path.join('/tmp', 'X_train_samples.npy')
        shutil.copyfile(X_samples_path, x_tmp_file)
        X_samples = np_load_file_auto_mmap(x_tmp_file)

        y_tmp_file = os.path.join('/tmp', 'y_samples.npy')
        shutil.copyfile(Y_samples_path, y_tmp_file)
        Y_samples = np_load_file_auto_mmap(y_tmp_file)

        has_samples = True
    except Exception:
        pass

    return has_samples, X_samples, Y_samples

def load_samples(dir_path):
    X_samples = None
    Y_samples = None
    has_samples = False
    try:
        x_tmp_file = os.path.join(dir_path, 'X_train_samples.npy')
        X_samples = np_load_file_auto_mmap(x_tmp_file)

        y_tmp_file = os.path.join(dir_path, 'y_samples.npy')
        Y_samples = np_load_file_auto_mmap(y_tmp_file)

        has_samples = True
    except Exception:
        pass

    return has_samples, X_samples, Y_samples

def get_datasets(X_train, Y_train, X_test, Y_test, has_samples, X_samples, Y_samples,
                 mode, classes, reshape_to, X_train_raw=None, online_dsp_config=None,
                 augmentation_enabled=False, object_detection_last_layer: Optional[ObjectDetectionLastLayer]=None,
                 object_detection_batch_size=None, ensure_determinism=False):

    # Autotune parallel calls is usually sensible, but can be non-deterministic, so we
    # allow disabling for integration tests to prevent intermittent fails.
    parallel_calls_policy = None if ensure_determinism else tf.data.experimental.AUTOTUNE

    if mode == 'object-detection':
        def format_object_detection_data(target_shape):
            def mapper(image, label):
                """Get image into the correct format for the training scripts"""
                reshaped = tf.reshape(image, target_shape)
                boxes = label[0]
                classes = label[1]
                return reshaped, (boxes, classes)
            return mapper

        # derive width/height of dataset from passed reshape_to
        height, width, num_channels = reshape_to
        if num_channels not in [1, 3]:
            raise Exception(f"Only single channel, or RGB images are supported")

        train_dataset = get_dataset_object_detection(X_train, width, height, num_channels,
             Y_train, classes, augment=augmentation_enabled)
        validation_dataset = get_dataset_object_detection(X_test, width, height, num_channels,
             Y_test, classes, augment=False)

        if object_detection_last_layer == 'fomo':
            target_shape = reshape_to
        else:
            # The Google obj det framework expects an extra dimension here
            target_shape = (1, *reshape_to)

        train_dataset = train_dataset.map(format_object_detection_data(target_shape),
                                          parallel_calls_policy)
        validation_dataset = validation_dataset.map(format_object_detection_data(target_shape),
                                          parallel_calls_policy)

        # Cache datasets in memory
        if not augmentation_enabled and ei_tensorflow.utils.can_cache_data(X_train):
            train_dataset = train_dataset.cache()
            validation_dataset = validation_dataset.cache()

        # For SSD models we don't have expert mode, so we pass batch size via input and set batch size here.
        # FOMO sets batch size in expert mode.
        if object_detection_last_layer != 'fomo':
            # TODO: Add shuffle here for SSD models. This was causing a memory leak.
            print(f'Using batch size: {object_detection_batch_size}', flush=True)
            train_dataset = train_dataset.batch(object_detection_batch_size, drop_remainder=False)
            validation_dataset = validation_dataset.batch(object_detection_batch_size, drop_remainder=False)

        return train_dataset, validation_dataset, None
    else:
        if X_train_raw is None:
            train_dataset = get_dataset_standard(X_train, Y_train)
        else:
            train_dataset = get_dataset_standard(X_train_raw, Y_train)
            train_dataset = train_dataset.map(get_dsp_function(online_dsp_config), parallel_calls_policy)
        validation_dataset = get_dataset_standard(X_test, Y_test)
        if has_samples:
            samples_dataset = get_dataset_standard(X_samples, Y_samples)
        else:
            samples_dataset = None

        # Reshape data on the fly, using multiprocessing if possible
        train_dataset = train_dataset.map(get_reshape_function(reshape_to), parallel_calls_policy)
        validation_dataset = validation_dataset.map(get_reshape_function(reshape_to), parallel_calls_policy)
        if has_samples:
            samples_dataset = samples_dataset.map(get_reshape_function(reshape_to), parallel_calls_policy)

        # Cache datasets in memory
        if ei_tensorflow.utils.can_cache_data(X_train):
            train_dataset = train_dataset.cache()
            validation_dataset = validation_dataset.cache()

        return train_dataset, validation_dataset, samples_dataset

def get_dsp_function(online_dsp_config):
    # This assumes an Edge Impulse DSP implementation has been made available,
    # for example by being copied into the filesystem in the train() method of learn-block-keras.ts.
    from dsp import generate_features
    # Logic based on createFeatures in templates.ts
    def run_dsp(sample_outer, label):
        def run(sample):
            freq = 0
            if online_dsp_config['input_type'] == 'time-series':
                # For time series data the interval is in the first column
                freq = ei_tensorflow.utils.calculate_freq(sample[0])
                data = sample[1:]
            else:
                data = sample
            result = generate_features(online_dsp_config['implementation_version'], False, data,
                                       online_dsp_config['axes'], freq, **online_dsp_config['params'])
            features = result['features']
            return np.array(features, np.float32)

        run_result = tf.numpy_function(run, [sample_outer], tf.float32)
        return run_result, label

    return run_dsp

def get_callbacks(dir_path, mode, best_model_path, object_detection_last_layer: ObjectDetectionLastLayer,
                  is_enterprise_project, max_training_time_s, max_gpu_time_s, enable_tensorboard):
    callbacks = []
    if mode == 'object-detection':
        if (object_detection_last_layer == 'fomo'):
            handle_training_deadline_callback = HandleTrainingDeadline(
                is_enterprise_project=is_enterprise_project, max_training_time_s=max_training_time_s,
                max_gpu_time_s=max_gpu_time_s)
            callbacks.append(handle_training_deadline_callback)
    else:
        # Saves the best model, based on validation loss (hopefully more meaningful than just accuracy)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(best_model_path,
            monitor='val_loss',
            save_best_only=True, mode='auto',
            # It's important to save and load the whole model and not just the weights because,
            # if we do any fine tuning during transfer learning, the fine tuned model has a
            # slightly different data structure.
            save_weights_only=False,
            verbose=0)

        handle_training_deadline_callback = HandleTrainingDeadline(
            is_enterprise_project=is_enterprise_project, max_training_time_s=max_training_time_s,
            max_gpu_time_s=max_gpu_time_s)

        # We'll pass this array into the train function and add more callbacks there
        callbacks.append(model_checkpoint_callback)
        callbacks.append(handle_training_deadline_callback)

    if enable_tensorboard:
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(dir_path, 'tensorboard_logs'),
                                                     # Profile batches 1-100
                                                     profile_batch=(1,101))
        callbacks.append(tb_callback)

    if not os.path.exists(os.path.join(dir_path, 'artifacts')):
        os.makedirs(os.path.join(dir_path, 'artifacts'))
    callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(dir_path, 'artifacts', 'training_log.csv')))

    return callbacks

# Loads the best model from disk
def load_best_model(best_model_path, akida_model=False):
    if akida_model:
        import cnn2snn
        return cnn2snn.load_quantized_model(best_model_path)
    # Includes workaround for https://github.com/edgeimpulse/edgeimpulse/issues/1419
    model = tf.keras.models.load_model(best_model_path, compile=False)
    model.compile()
    return model

def replace_layers(model, layer_to_remove, layer_to_substitute, substitute_args=()):
    """Replaces layers in a model
    """
    def clone_function(layer):
        if isinstance(layer, layer_to_remove):
            # Replace unwanted layers with an empty Layer(), which is an identity function
            return layer_to_substitute(*substitute_args)
        else:
            return layer.__class__.from_config(layer.get_config())

    clean_model = tf.keras.models.clone_model(model, input_tensors=None,
                                              clone_function=clone_function)
    weights = model.get_weights()
    clean_model.set_weights(weights)
    clean_model.compile()
    return clean_model

def remove_dropout(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            print('Removing Dropout layers from model.', flush=True)
            return replace_layers(model, tf.keras.layers.Dropout, tf.keras.layers.Dropout, [0])
    return model

def clean_model_for_syntiant(model):
    """Syntiant TDK uses Python 3.6/TF 2.3.2. There's an incompatibility when trying to load
       Gaussian layers during posterior search. To avoid this, for Syntiant targets we
       remove any Gaussian layers before saving the model. They only have an impact at training
       time, so this is OK.

    Args:
        model: A Keras model instance
    """
    return replace_layers(model, tf.keras.layers.GaussianNoise, tf.keras.layers.Layer)

def save_model(keras_model, best_model_path, dir_path, saved_model_dir,
               h5_model_path, syntiant_target=False, akida_model=False):
    saved_model_complete = False

    def best_performing_thread():
        time.sleep(2)
        while not saved_model_complete:
            print('Still saving model...', flush=True)
            time.sleep(5)

    progress_thread = threading.Thread(target=best_performing_thread)
    progress_thread.start()

    try:
        # If we have a best model checkpoint, we should load it, replacing
        # whatever happens to be in memory. If not (which may happen if the user
        # has legacy expert mode code before we added the 'callbacks' array) then
        # just use the original model
        if os.path.exists(best_model_path):
            print('Saving best performing model... (based on validation loss)', flush=True)
            keras_model = load_best_model(best_model_path, akida_model=akida_model)
        else:
            print('Saving model...', flush=True)

        if syntiant_target:
            keras_model = clean_model_for_syntiant(keras_model)

        # we need to explicitly set the input_shape on the `save` call on the Keras model
        # as we have save_traces=False - otherwise e.g. ONNX has no idea what the input
        # to the model is
        input_shape = keras_model.layers[0].get_input_at(0).get_shape()
        input_shape_tuple = tuple(list(input_shape)[1:])

        # Save the model to disk and zip it for download.
        # Save both TF savedmodel and keras h5 for maximum compatibility.
        saved_model_path = os.path.join(dir_path, saved_model_dir)
        keras_model.save(saved_model_path, save_format='tf', save_traces=False,
            signatures=get_concrete_function(keras_model, input_shape_tuple))
        shutil.make_archive(os.path.join(dir_path, saved_model_dir),
                            'zip', root_dir=dir_path,
                            base_dir=saved_model_dir)
        h5_path = os.path.join(dir_path, h5_model_path)
        keras_model.save(h5_path, save_format='h5')
        with zipfile.ZipFile(h5_path + '.zip', "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(h5_path, os.path.basename(h5_path))
        os.remove(h5_path)

    finally:
        saved_model_complete = True

    if os.path.exists(best_model_path):
        print('Saving best performing model OK', flush=True)
    else:
        print('Saving model OK', flush=True)
    print('', flush=True)

    return keras_model

# Utility function for saving an image out of a tf.data.Dataset
def save_sample_image(dir_path, dataset):
    for image, label in dataset.take(1):
        encoded = tf.io.encode_png(tf.image.convert_image_dtype(image, tf.uint16))
        tf.io.write_file(os.path.join(dir_path, 'sample.png'), encoded)

# Get class weights for auto-balance
def get_class_weights(Y_train):
    # y_train is one_hot labels so we can use the indexes of where the
    # 1.0 is set to derive the class label.
    _rows, idxs = np.where(Y_train==1.0)
    class_weights = dict(Counter(idxs))
    total_count = sum(class_weights.values())

    for c in class_weights.keys():
        class_weights[c] = total_count / class_weights[c] / len(class_weights.keys())

    return class_weights

def get_friendly_time(total_length_s):
    hours = math.floor(total_length_s / 3600)
    total_length_s -= (hours * 3600)
    minutes = math.floor(total_length_s / 60)
    total_length_s -= (minutes * 60)
    seconds = math.floor(total_length_s)

    tt = ''
    if (hours > 0):
        tt = tt + str(hours) + 'h '
    if (hours > 0 or minutes > 0):
        tt = tt + str(minutes) + 'm '
    tt = tt + str(seconds) + 's '
    return tt.strip()

def print_training_time_exceeded(is_enterprise_project, max_training_time_s, total_time):
    print('')
    print('ERR: Estimated training time (' + get_friendly_time(total_time) + ') ' +
        'is larger than compute time limit (' + get_friendly_time(max_training_time_s) + ').')
    print('')
    if (is_enterprise_project):
        print('You can up the compute time limit under **Dashboard > Performance settings**')
    else:
        print('See https://docs.edgeimpulse.com/docs/tips-and-tricks/lower-compute-time on tips to lower your compute time requirements.')
        print('')
        print('Alternatively, the enterprise version of Edge Impulse has no limits, see ' +
            'https://www.edgeimpulse.com/pricing for more information.');

def check_gpu_time_exceeded(max_gpu_time_s, total_time):
    # Check we have a limit on GPU time
    if (max_gpu_time_s == None):
        return

    # Check we're running on GPU
    device_count = ei_tensorflow.gpu.get_gpu_count()
    if (device_count == 0):
        return

    # Allow some tolerance
    tolerance = 1.2
    if (max_gpu_time_s * tolerance > total_time):
        return

    # Show an error message
    print('')
    print('ERR: Estimated training time (' + get_friendly_time(total_time) + ') ' +
        'is greater than remaining GPU compute time limit (' + get_friendly_time(max_gpu_time_s) + ').')
    print('Try switching to CPU for training, or contact sales (hello@edgeimpulse.com) to ' +
        'increase your GPU compute time limit.')
    print('')

    # End the job
    exit(1)

class HandleTrainingDeadline(Callback):
    """ Check when we run out of training time. """

    def __init__(self, max_training_time_s: float, max_gpu_time_s: float, is_enterprise_project: bool):
        self.max_training_time_s = max_training_time_s
        self.max_gpu_time_s = max_gpu_time_s
        self.is_enterprise_project = is_enterprise_project
        self.epoch_0_begin = time.time()
        self.epoch_1_begin = time.time()
        self.printed_est_time = False

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch == 0):
            self.epoch_0_begin = time.time()
        if (epoch == 1):
            self.epoch_1_begin = time.time()

    def on_epoch_end(self, epoch, logs):
        # on both epoch 0 and epoch 1 we want to estimate training time
        # if either is above the training time limit, then we exit
        if (epoch == 0 or epoch == 1):
            time_per_epoch_ms = 0
            if (epoch == 0):
                time_per_epoch_ms = float(time.time() - self.epoch_0_begin) * 1000
            elif (epoch == 1):
                time_per_epoch_ms = float(time.time() - self.epoch_1_begin) * 1000

            total_time = time_per_epoch_ms * self.params['epochs'] / 1000

            # uncomment this to debug the training time algo:
            # print('Epoch', epoch, '- time for this epoch: ' + get_friendly_time(time_per_epoch_ms / 1000) +
            #     ', estimated training time:', get_friendly_time(total_time))

            if (total_time > self.max_training_time_s * 1.2):
                print_training_time_exceeded(self.is_enterprise_project, self.max_training_time_s, total_time)
                exit(1)
            check_gpu_time_exceeded(self.max_gpu_time_s, total_time)

def get_concrete_function(keras_model, input_shape):
    # To produce an optimized model, the converter needs to see a static batch dimension.
    # At this point our model has an unspecified batch dimension, so we need to set it to 1.
    # See: https://github.com/tensorflow/tensorflow/issues/42286#issuecomment-681183961
    input_shape_with_batch = (1,) + input_shape
    run_model = tf.function(lambda x: keras_model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec(input_shape_with_batch, keras_model.inputs[0].dtype))
    return concrete_func
