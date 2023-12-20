import shutil
from pathlib import Path
from typing import Tuple, Optional
import os, json

import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .translate import translate_function

from ei_tensorflow import conversion
from ei_shared.pretrained_weights import get_or_download_pretrained_weights
from ei_shared.pretrained_weights import get_weights_path_if_available

import numpy as np
from jax import lax, vmap, jit
import jax.numpy as jnp

WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())

def _mobile_net_trunk_imagenet_96_weights(num_channels: int, alpha: float):
    # note: regardless of what resolution we intend to use for actual image
    #  input we emperically get the best result for anomaly detection from
    #  using 96x96 imagenet weights. i (mat) suspect this is due to fact the
    #  anomaly detection features are usually not large, so the lower the
    #  resolution of the pretrained weights the better.

    # Define the allowed combinations of num_channels and alpha for the pretrained weights
    allowed_combinations = [{'num_channels': 1, 'alpha': 0.1},
                            {'num_channels': 1, 'alpha': 0.35},
                            {'num_channels': 3, 'alpha': 0.1},
                            {'num_channels': 3, 'alpha': 0.35},
                            {'num_channels': 3, 'alpha': 1.0}]

    weights = get_or_download_pretrained_weights(WEIGHTS_PREFIX, num_channels, alpha, allowed_combinations)

    mobile_net_v2 = MobileNetV2(input_shape=(96, 96, num_channels),
                                weights=weights,
                                alpha=alpha, include_top=True)
    cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
    mobile_net_trunk = Model(inputs=mobile_net_v2.input, outputs=cut_point.output)
    return mobile_net_trunk.get_weights()

class MobileNetFeatureExtractor(object):

    def __init__(self,
                 input_shape: Tuple[int],
                 use_mobile_net_pretrained_weights: bool,
                 mobile_net_alpha: int,
                 seed: int):
        """ Mobile Net Feature Extractor
        Args:
            input_shape: (H,W,C) shape of expected input. Used to build
                MobileNet trunk.
            use_mobile_net_pretrained_weights: if true initialise MobileNet
                with ImageNet weights for 96x96 input. We use 96x96 weights
                since we'll only being used the start of mobilenet to reduce
                to 1/8th input.
            mobile_net_alpha: what alpha to use for the MobileNet
        """

        # validate input shape
        if len(input_shape) != 3:
            raise Exception(f"Expected input_shape be (H,W,C), not {input_shape}")
        _img_height, _img_width, num_channels = input_shape
        if num_channels not in [1, 3]:
            raise Exception(f"VisualAD only supports num_channels of 1 or 3,"
                            f" not {num_channels} from input_shape {input_shape}")
        self.input_shape = input_shape

        # construct mobile net trunk.
        # we create the entire mobilenet_v2 of whatever input_shape size has been
        # requested and then truncate to the 'block_6_expand_relu' layer.
        if seed is not None:
            tf.random.set_seed(seed)

        # set initial weights to use for the feature extractor.
        # explicitly specify the container path to these weights if available, as direct
        # download will be blocked when training runs from Studio in production
        # If imagenet weights are being loaded, alpha can be one of
        # `0.35`, `0.50`, `0.75`, `1.0`, `1.3` or `1.4` only.
        imagenet_supported = (num_channels==3 and mobile_net_alpha in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4])
        initial_weights = None
        if imagenet_supported and not use_mobile_net_pretrained_weights:
            # it might be that the input size is different to the standards model sizes
            # so choose the closest.
            availiable_dims = [96, 128, 160, 192, 224]
            width_height_dim = input_shape[0]
            initial_weights_dim = min(availiable_dims, key=lambda val: abs(val - width_height_dim))
            initial_weights = get_weights_path_if_available(WEIGHTS_PREFIX, num_channels,
                                                            mobile_net_alpha, initial_weights_dim)
            # use 'imagenet' weights as backup
            initial_weights = 'imagenet' if initial_weights is None else initial_weights

        if initial_weights is not None:
            print("Using imagenet as initial weights for the feature extractor")
        mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                    weights=initial_weights, alpha=mobile_net_alpha, include_top=False)
        cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
        self.mobile_net_trunk = Model(inputs=mobile_net_v2.input,
                                      outputs=cut_point.output)

        # only load in 96x96 imagenet weights if requested.
        # note: we couldn't do this during the above creation of the network since
        # the user might be working with a network input larger than 96x96 and keras
        # doesn't support creating a network with one resolution, but using weights
        # from another resolution. However, because the network is fully convolutional
        # we _can_ load the weights from a 96x96 network _after_ creation. we use
        # these small 96x96 weights, regardless of the actual input resolution, since
        # we expect features related to anomalies to be smaller than the entire image.
        if use_mobile_net_pretrained_weights:
            self.mobile_net_trunk.set_weights(
                _mobile_net_trunk_imagenet_96_weights(
                    num_channels, mobile_net_alpha))

        # TODO: if num_channels==1 and not use_mobile_net_pretrained_weights
        #       then at this point, the mobile net is randomly init'd... and
        #       i don't know that _ever_ gives a good result...

    def extract_features(self, x: np.ndarray):
        _batch, img_height, img_width, num_channels = x.shape
        if self.input_shape != (img_height, img_width, num_channels):
            raise Exception(f"Expected input to be batched {self.input_shape}"
                            f" not {x.shape}")
        return self._batch_run(x)

    def _batch_run(self, x: np.ndarray, batch_size: int=64):
        # TODO(mat) will only have to do these during training, not inference
        if len(x) < batch_size:
            return self.mobile_net_trunk(x).numpy()
        idx = 0
        y = []
        while idx < len(x):
            x_batch = x[idx:idx+batch_size]
            y.append(self.mobile_net_trunk(x_batch).numpy())
            idx += batch_size
        return np.concatenate(y)

    @property
    def output_shape(self):
        return self.mobile_net_trunk.output_shape

    def save_model(self, dir_path: str, saved_model_dir: str, representative_data: tf.data.Dataset):
        h5_model = Path(dir_path) / 'model.h5'
        tflite_float32 = Path(dir_path) / 'model.tflite'
        tflite_int8 = Path(dir_path) / 'model_quantized_int8_io.tflite'

        # Is not allowed to be None, so using a dummy filename as not relevant for visual GMM (or?)
        BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'no_best_model.hdf5')

        models = conversion.convert_to_tf_lite(model=self.mobile_net_trunk,
                                               best_model_path=BEST_MODEL_PATH,
                                               dir_path=dir_path,
                                               saved_model_dir=saved_model_dir,
                                               h5_model_path=h5_model,
                                               validation_dataset=representative_data,
                                               model_input_shape=self.input_shape,
                                               model_filenames_float=tflite_float32,
                                               model_filenames_quantised_int8=tflite_int8)

        return models

class SpatialAwareRandomProjection(object):

    def __init__(self,
                 random_projection_dim: int,
                 seed: int):
        self.random_projection_dim = random_projection_dim
        self.seed = seed
        self.fit_and_project_called = False

    def fit_and_project(self, x: np.ndarray):
        # record details of the shapes of x; specifically
        # the spatial component of the shape (i.e. everything but the last
        # dimension). we do this since we're going to have to flatten x
        # to be able to run through the sklearn projection which only
        # supports 2D data.
        spatial_shape = x.shape[:-1]
        x_dimension = x.shape[-1]

        # convert from, say, (num_instances=10, height=3, width=3,
        # n_features=96) to flattened (90, 96)
        flat_x = x.reshape((-1, x_dimension))

        # "fit" the projection (which, for this, just creates the
        # projection matrix)
        self.random_projection = GaussianRandomProjection(
                n_components=self.random_projection_dim,
                random_state=self.seed)
        self.random_projection.fit(flat_x)

        # apply the projection which will go from, say, (90, 96) -> (90, 8) if
        # random_projection_dim=8
        flat_x = self.random_projection.transform(flat_x).astype(np.float32)

        # restore the original spatial shape; e.g. (90, 8) -> (10, 3, 3, 8)
        projected_x = flat_x.reshape((*spatial_shape, self.random_projection_dim))
        self.fit_and_project_called = True
        return projected_x

    def project(self, y: np.ndarray, use_jax: bool=False):
        if not self.fit_and_project_called:
            raise Exception("Must call fit_and_project() before project()")
        if use_jax:
            project_fn = translate_function(
                self.random_projection, GaussianRandomProjection.transform)
            spatial_project_fn = vmap(vmap(project_fn))
            return spatial_project_fn(y)
        else:
            spatial_shape = y.shape[:-1]
            y_dimension = y.shape[-1]
            flat_y = y.reshape((-1, y_dimension))
            flat_y = self.random_projection.transform(flat_y).astype(np.float32)
            return flat_y.reshape((*spatial_shape, self.random_projection_dim))


class AveragePooling(object):
    # minimal port of dh-haiku avg pool (so we don't need to pull in the
    # entire package just for this one piece of code)
    # see https://github.com/deepmind/dm-haiku/blob/ab16af8230b1be279cf99a660e0fe95bd759e977/haiku/_src/pool.py#L105
    # assumes x in 4d... TODO(mat) should we pull in _infer_shape too?

    def __init__(self, pool_size: int, pool_stride: int):
        self.pool_size = pool_size
        self.pool_stride = pool_stride

    def __call__(self, x: np.ndarray):
        window_shape = (1, self.pool_size, self.pool_size, 1)
        strides = (1, self.pool_stride, self.pool_stride, 1)
        padding = 'VALID'
        reduce_window_args = (0., lax.add, window_shape, strides, padding)
        pooled = lax.reduce_window(x, *reduce_window_args)
        return pooled / np.prod(window_shape)

class SpatialAwareGaussianMixtureAnomalyScorer(object):

    def __init__(self, n_components: int, seed: int):
        self.gmm = GaussianMixture(
            n_components=n_components, random_state=seed,
            covariance_type='full')
        self.scaler = StandardScaler()
        self.fit_called = False

    def fit(self, x: np.ndarray):
        # TODO(mat): pull this out into util, share with random projection
        # flat for GMM and scalar
        x_dimension = x.shape[-1]
        flat_x = x.reshape((-1, x_dimension))
        # fit GMM and score
        self.gmm.fit(flat_x)
        scores = self.gmm.score_samples(flat_x)
        # use scores to fit scalar
        # note: scalar requires trailing dimension
        scores = np.expand_dims(scores, axis=-1)
        self.scaler.fit(scores)
        self.fit_called = True

        # rerun scaler over these scores from training data
        # and record the suggested threshold value.
        scaled_scores = np.abs(self.scaler.transform(scores))
        self.nominal_threshold_score = np.max(scaled_scores)

        self.feature_dim = x_dimension

    def anomaly_score(self, x: np.ndarray, use_jax: bool=False):
        if not self.fit_called:
            raise Exception("Must call fit() before anomaly_score()")

        if use_jax:
            # for the jax versions we can compose a scalar version function as
            # gmm_score -> standardise -> absolute and then create a spatial
            # version with two vmaps.

            # convert inference functions
            gmm_score_fn = translate_function(
                self.gmm, GaussianMixture.score_samples)
            standardise_fn = translate_function(
                self.scaler, StandardScaler.transform)
            # stitch them together into one function (with absolute)
            def single_element_score_fn(x):
                scores = gmm_score_fn(x)
                scores = standardise_fn(scores)
                return jnp.abs(scores)
            # compile vectorised form for spatial version and return
            spatial_score_fn = vmap(vmap(single_element_score_fn))
            return spatial_score_fn(x)

        else:
            # for the non jax version we need to flatten the x before running
            # the gmm_score -> standardise -> absolute before restoring the
            # shape with a reshape.

            # flatten
            spatial_shape = x.shape[:-1]
            x_dimension = x.shape[-1]
            flat_x = x.reshape((-1, x_dimension))
            # score via GMM
            scores = self.gmm.score_samples(flat_x)
            # standardise with absolute value
            scores = np.expand_dims(scores, axis=-1)
            scores = self.scaler.transform(scores)
            scores = np.abs(scores)
            # return with restored spatial shape
            return scores.reshape(spatial_shape)

class VisualAnomalyDetection(object):

    def __init__(self,
                 input_shape: Tuple[int],
                 use_mobile_net_pretrained_weights: bool,
                 mobile_net_alpha: float,
                 random_projection_dim: int,
                 pool_size: int,
                 pool_stride: int,
                 gmm_n_components: int,
                 seed: int):
        """ Visual Anomaly Detection.
            input_shape: (H,W,C) shape of expected input. Used to build
                MobileNet trunk. see MobileNetFeatureExtractor.
            use_mobile_net_pretrained_weights: if true initialise MobileNet
                with ImageNet weights for 96x96 input. We use 96x96 weights
                since we'll only being used the start of mobilenet to reduce
                to 1/8th input. see MobileNetFeatureExtractor.
            mobile_net_alpha: what alpha to use for the MobileNet.
            random_projection_dim: projection dimension for spatially aware
                random projection to run on feature maps from mobilenet. if
                None no random projection is used.
            pool_size: pooling kernel size (square) for average pooling post
                random projection.
            pool_stride: pooling stride for average pooling post random
                projection.
            gmm_n_components: num components to pass to spatially aware mixture
                model for scoring
            seed: seed for random number generation.
            use_mobile_net_pretrained_weights: if true initialise MobileNet
                with ImageNet weights for 96x96 input. We use 96x96 weights
                since we'll only being used the start of mobilenet to reduce
                to 1/8th input. see MobileNetFeatureExtractor.
        """
        self.input_shape = input_shape
        self.feature_extractor = MobileNetFeatureExtractor(
            input_shape, use_mobile_net_pretrained_weights,
            mobile_net_alpha, seed)
        self.feature_map_shape = None
        if random_projection_dim is not None:
            self.random_projection = SpatialAwareRandomProjection(
                random_projection_dim, seed)
        else:
            self.random_projection = None
        self.avg_pooling = AveragePooling(pool_size, pool_stride)
        self.mixture_model = SpatialAwareGaussianMixtureAnomalyScorer(
            gmm_n_components, seed)

    def fit(self, x: np.ndarray):
        feature_map = self.feature_extractor.extract_features(x)
        if self.random_projection is not None:
            feature_map = self.random_projection.fit_and_project(feature_map)
        pooled_feature_map = self.avg_pooling(feature_map)
        self.mixture_model.fit(pooled_feature_map)

    def write_metadata(self, dir_path: str):
        metadata_filename = 'anomaly_metadata.json'

        with open(os.path.join(dir_path, metadata_filename), 'w') as f:
            f.write(json.dumps({
                'nominalThresholdScore': self.mixture_model.nominal_threshold_score
            }, indent=4))

    def reference_input_shape(self):
        if not self.mixture_model.fit_called:
            raise Exception("Must call fit() before input_shape()")

        return self.mixture_model.feature_dim

    def _save_scorer(self, dir_path: str, reduction_mode: Optional[str]):
        # There is only currently a float32 version of the model, since quantization
        # destroys performance. In the future we may produce an int8 version but it
        # will require quantization-aware training.
        scorer_tflite_float32_path = Path(dir_path) / 'scorer.float32.tflite'
        score_fn = self.spatial_anomaly_score_fn(reduction_mode=reduction_mode,
                                                 use_jax=True)
        output_shape_without_batch = self.feature_extractor.output_shape[1:]
        scorer_tflite_model = conversion.convert_jax_to_tflite_float32(
            jax_function=score_fn,
            input_shape=output_shape_without_batch
        )

        with open(scorer_tflite_float32_path, 'wb') as f:
            f.write(scorer_tflite_model)


    def save_model(self,
                   dir_path: str,
                   saved_model_dir: str,
                   representative_data: tf.data.Dataset,
                   reduction_mode: Optional[str]):

        model_path = Path(dir_path)
        _ = self.feature_extractor.save_model(dir_path, saved_model_dir, representative_data)

        self._save_scorer(dir_path, reduction_mode=reduction_mode)

        self.write_metadata(dir_path)

        return model_path

    def feature_extractor_fn(self):
        return self.feature_extractor.extract_features

    def feature_extractor_input_shape(self):
        return self.input_shape

    def spatial_anomaly_score_fn(self,
                                 reduction_mode: Optional[str],
                                 use_jax: bool=False):
        def score_fn(feature_map):
            if self.random_projection is not None:
                feature_map = self.random_projection.project(
                    feature_map, use_jax=use_jax)
            pooled_feature_map = self.avg_pooling(feature_map)
            spatial_scores = self.mixture_model.anomaly_score(
                pooled_feature_map, use_jax=use_jax)
            if reduction_mode is None:
                return spatial_scores
            elif reduction_mode == 'mean':
                return spatial_scores.mean(axis=(-1, -2))
            elif reduction_mode == 'max':
                return spatial_scores.max(axis=(-1, -2))
            else:
                raise Exception(f"Invalid reduction_mode [{reduction_mode}], expected [None, mean, max]")

        # if using jax, jit compile here
        if use_jax:
            score_fn = jit(score_fn)

        return score_fn

    def score(self, x: np.ndarray,
                reduction_mode: Optional[str]=None,
                use_jax: bool=False,
                batch_size: int=64):

        spatial_anomaly_score_fn = self.spatial_anomaly_score_fn(
            reduction_mode, use_jax
        )

        # for very large x, e.g. benchmarking, we need to batch the score_fn
        if batch_size is None:
            feature_map = self.feature_extractor.extract_features(x)
            scores = spatial_anomaly_score_fn(feature_map)
            return np.array(scores)
        else:
            idx = 0
            scores = []
            n_batches = 0
            while idx < len(x):
                x_batch = x[idx:idx+batch_size]
                feature_map = self.feature_extractor.extract_features(x_batch)
                batch_scores = spatial_anomaly_score_fn(feature_map)
                scores.append(np.array(batch_scores))
                idx += batch_size
                n_batches += 1
            return np.concatenate(scores)
