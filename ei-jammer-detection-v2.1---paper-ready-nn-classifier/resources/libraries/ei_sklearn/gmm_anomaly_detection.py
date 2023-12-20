from typing import Tuple

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import ei_tensorflow.inference

import numpy as np
import pickle
import json
import os
import datetime

class GaussianMixtureAnomalyScorer(object):

    def __init__(self, n_components: int = None, seed: int = None, options: dict = None):

        if isinstance(options, dict):
            seed = int(options['seed'])
            n_components = int(options['clusterCount'])

        if n_components is None and seed is None:
            raise Exception("Must provide GaussianMixtureAnomalyScorer with arguments")

        self.gmm = GaussianMixture(
            n_components=n_components, random_state=seed,
            covariance_type='full')
        self.scaler = StandardScaler()
        self.fit_called = False

    def run_training(self, dir_path: str, features_filename: str, axes: list):

        X = np.load(os.path.join(dir_path, features_filename))
        X = self.__class__.extract_axis(X, axes)

        self.fit(X)

    def save_model(self, dir_path: str, model_filename_tflite: str, model_filename_pickle: str,
        features_filename: str, metadata_filename: str, gmm_metadata_filename: str, axes: list):

        from ei_tensorflow.conversion import (convert_jax_to_tflite_float32)

        # save model as pickled
        self.pickle_gmm(dir_path, model_filename_pickle)

        # score training data to derive default threshold
        X = np.load(os.path.join(dir_path, features_filename))
        X = self.__class__.extract_axis(X, axes)
        scores = self.gmm.score_samples(X)
        suggested_threshold = max(0.1, np.percentile(scores, 99))

        self.save_gmm_attributes(dir_path, gmm_metadata_filename)
        self.write_metadata(dir_path, metadata_filename, axes, suggested_threshold)

        # also convert to tflite and save
        def jax_score(x):
            return self.score(x, use_jax=True)

        tflite_model = convert_jax_to_tflite_float32(
            jax_function=jax_score,
            input_shape=self.reference_input_shape(),
            redirect_streams=False)

        with open(os.path.join(dir_path, model_filename_tflite), 'wb') as f:
            f.write(tflite_model)

    def save_gmm_attributes(self, dir_path: str, output_filename: str = None):
        gmm_attributes = {
            'means': self.gmm.means_.tolist(),
            'covariances': self.gmm.covariances_.tolist(),
            'weights': self.gmm.weights_.tolist(),
        }
        if output_filename:
            with open(os.path.join(dir_path, output_filename), 'w') as f:
                f.write(json.dumps(gmm_attributes))
        else:
            print('Begin output')
            print(json.dumps(gmm_attributes))
            print('End output')

    def predict_samples(self, dir_path: str, features_filename: str,
        axes: list, output_filename: str = None):

        X = np.load(os.path.join(dir_path, features_filename))
        X = GaussianMixtureAnomalyScorer.extract_axis(X, axes)

        scores = self.gmm.score_samples(X)

        if output_filename:
            with open(os.path.join(dir_path, output_filename), 'w') as f:
                f.write(json.dumps(scores.tolist()))
        else:
            print('Begin output')
            print(json.dumps(scores.tolist()))
            print('End output')

    @staticmethod
    def run_classify_job(dir_path: str, model_filename: str, input_filename: str, axes: list,
        output_filename: str = None):

        # model should run with tflite runtime
        interpreter = ei_tensorflow.inference.prepare_interpreter(dir_path, model_filename)

        def detect_anomaly(inputs):
            # convert to float32 and run inference
            scores_tflite = []
            for item in inputs.astype(np.float32):
                X_sample = np.take(item, axes)
                score = ei_tensorflow.inference.run_model(mode='anomaly-gmm', interpreter=interpreter, item=X_sample, specific_input_shape=None)
                if len(score) == 0:
                    raise Exception("Format of score result was not a list as expected")
                scores_tflite.append(score[0])
            return scores_tflite

        input = np.load(input_filename)
        if (not isinstance(input[0], (np.ndarray))):
            input = np.array([ input ])

        if output_filename:
            with open(output_filename, 'w') as f:
                f.write(json.dumps(detect_anomaly(input)))
        else:
            print('Begin output')
            print(json.dumps(detect_anomaly(input)))
            print('End output')

    def write_metadata(self, dir_path: str, metadata_filename: str, axes: list, suggested_threshold: float):

        with open(os.path.join(dir_path, metadata_filename), 'w') as f:
            f.write(json.dumps({
                'created': datetime.datetime.now().isoformat(),
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'axes': axes,
                'defaultMinimumConfidenceRating': suggested_threshold
            }, indent=4))

    def pickle_gmm(self, dir_path: str, model_filename_pickle: str):

        with open(os.path.join(dir_path, model_filename_pickle), 'wb') as f:
            pickle.dump(self.gmm, f)

    def fit(self, x: np.array):
        # fit GMM
        self.gmm.fit(x)
        scores = self.gmm.score_samples(x)
        # use scores to fit scalar
        # note: scalar requires trailing dimension
        scores = np.expand_dims(scores, axis=-1)
        self.scaler.fit(scores)
        # record reference feature shape to be used in tflite conversion
        self.feature_dim = x.shape[-1]
        self.fit_called = True

    def reference_input_shape(self):
        if not self.fit_called:
            raise Exception("Must call fit() before input_shape()")
        return (self.feature_dim, )

    def score(self, x: np.array, use_jax: bool):
        if not self.fit_called:
            raise Exception("Must call fit() before score()")
        if use_jax:
            from .translate import translate_function
            import jax.numpy as jnp

            gmm_score_fn = translate_function(
                self.gmm, GaussianMixture.score_samples)
            scores = gmm_score_fn(x)
            # note: we add trailing dimension ONLY to match shape of non
            # jax version
            scores = jnp.expand_dims(scores, axis=-1)
            standardise_fn = translate_function(
                self.scaler, StandardScaler.transform)
            scores = standardise_fn(scores)
            return jnp.abs(scores)
        else:
            scores = self.gmm.score_samples(x)
            # recall: scalar requires trailing dimension
            scores = np.expand_dims(scores, axis=-1)
            scores = self.scaler.transform(scores)
            return np.abs(scores)

    @staticmethod
    def extract_axis(x: np.array, axes: list):
        return np.take(x, axes, axis=1)
