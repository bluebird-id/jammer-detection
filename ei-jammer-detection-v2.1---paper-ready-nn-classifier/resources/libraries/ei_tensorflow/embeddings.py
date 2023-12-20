import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shutil, datetime, json, time, threading, sys, os, math
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten
from tensorflow.keras.models import Sequential, Model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# This creates NN embeddings, will use tSNE if <=5000 samples, or PCA if larger than 5000
def create_embeddings(base_model, dir_path, out_file_x):
    start_time = time.time()

    print('Creating embeddings...')

    try:
        SHAPE = tuple(base_model.layers[0].get_input_at(0).get_shape().as_list()[1:])

        x_file = os.path.join(dir_path, 'X_train_features.npy')

        # always read through mmap
        rows = None
        with open(x_file, 'rb') as npy:
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            rows = shape[0]

        model = Sequential()
        model.add(InputLayer(input_shape=SHAPE, name='x_input'))
        model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output))
        model.add(Flatten())

        X_pred = pred_from_savedmodel(model, SHAPE, rows, x_file)

        if (rows > 5000):
            print('WARN: More than 5000 samples, using PCA to create embeddings.')
            scaler = StandardScaler()
            X_pred = scaler.fit_transform(X_pred)

            pca = PCA(n_components=2, random_state=3)
            dr_res = pca.fit_transform(X_pred)
        else:
            tsne = TSNE(2, learning_rate='auto', init='pca')
            dr_res = tsne.fit_transform(X_pred)

        np.save(out_file_x, np.ascontiguousarray(dr_res))

        time_s = round(time.time() - start_time)
        second_string = 'second' if time_s == 1 else 'seconds'
        print('Creating embeddings OK (took ' + str(time_s) + ' ' + second_string + ')')
        print('')
    except Exception as e:
        print('WARN: Creating embeddings failed:', e)
        print('')

def time_ms():
    return round(time.time() * 1000)

def pred_from_savedmodel(model, SHAPE, sample_count, x_path):
    last_update = 0
    X = np.load(x_path, mmap_mode='r')

    # read first el so we can check what the shape of the embedding will be (need to know the size of the X_pred beforehand)
    X_0 = X[0:1]
    X_0 = X_0.reshape(tuple([ X_0.shape[0] ]) + SHAPE)

    embeddings_len = model.predict(X_0, verbose=0).shape[1]
    X_pred = np.memmap('/tmp/X_pred.npy', dtype='float32', mode='w+', shape=(sample_count, embeddings_len))

    slice_size = 100
    slice_count = math.ceil(sample_count / slice_size)
    for i in np.arange(0, sample_count, slice_size):
        begin_pred = time_ms()

        # If we don't re-open the file here the underlying structure keeps caching all data ?!#@
        X = np.load(x_path, mmap_mode='r')

        X_slice = X[i:i+slice_size].copy()
        X_slice = X_slice.reshape(tuple([ X_slice.shape[0] ]) + SHAPE)
        X_pred[i:i+slice_size] = model.predict(X_slice, verbose=0)

        end_pred = time_ms()
        if i == 0:
            eta = (end_pred - begin_pred) * slice_count
            if (eta > 10000):
                raise Exception('Estimated time to create embeddings is larger than 10 sec., skipping (estimated: ' + str(eta) + ' ms.)')

        if (time_ms() - last_update > 3000):
            print('[' + str(i).rjust(len(str(sample_count))) + '/' + str(sample_count) + '] Creating embeddings...')
            last_update = time_ms()

    print('[' + str(sample_count) + '/' + str(sample_count) + '] Creating embeddings...')

    return X_pred
