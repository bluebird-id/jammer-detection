import sys, os, json
import logging
import warnings

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable annoying warning when GPU not found https://github.com/google/jax/issues/6805
logging.getLogger("absl").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

sys.path.append('./resources/libraries')

from ei_sklearn.visual_anomaly_detection import VisualAnomalyDetection
import numpy as np

# Suppress Numpy deprecation warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# Filter out this erroneous warning (https://stackoverflow.com/a/70268806 for context)
warnings.filterwarnings('ignore', 'Custom mask layers require a config and must override get_config')

def ei_log(msg: str):
    print("EI_LOG_LEVEL=info", msg)

def enumerated_n_tiles(a, n=31, as_int=False):
    p = np.percentile(a, np.linspace(0, 100, n))
    if as_int:
        p = p.astype(int)
    return list(enumerate(p))

print('Loading training set...')

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'options.json')) as f:
    options = json.load(f)

features_filename = 'X_train_features.npy'
saved_model_dir = options['savedModelDir']
height_width = int(options['heightWidth'])
num_channels = int(options['numChannels'])
gmm_n_components = int(options['numComponents'])
gmm_random_projection_dim = int(options['randomProjectionDim'])
mobile_net_alpha = float(options['mobileNetAlpha'])

# Load training data and reshape the array
X_train = np.load(os.path.join(dir_path, features_filename))
new_shape = (X_train.shape[0], height_width, height_width, num_channels)
X_train = np.reshape(X_train, new_shape)

print('Loading training set OK')
print('Training model...')
print('Training on {0} inputs'.format(len(X_train)))

# construct visual-AD and fit
visual_ad = VisualAnomalyDetection(
    input_shape=(height_width, height_width, num_channels),
    use_mobile_net_pretrained_weights=True,
    mobile_net_alpha=mobile_net_alpha,
    random_projection_dim=gmm_random_projection_dim,
    pool_size=3,
    pool_stride=2,
    gmm_n_components=gmm_n_components,
    seed=123
)
visual_ad.fit(X_train)


# rerun over training dataset to get a distribution of training values for different reduction modes
# this is introduced for release debugging, but may be useful longer term
LOG_SCORE_PERCENTILES = False
if LOG_SCORE_PERCENTILES:
    for reduction_mode in [None, 'mean', 'max']:
        training_scores = visual_ad.score(X_train, reduction_mode=reduction_mode, use_jax=True).astype(np.float32)
        # recall; None produces (N,H,W) whereas mean and max produce just (N,)
        score_percentiles = enumerated_n_tiles(training_scores, as_int=True)

        ei_log(f"training score ntiles reduction_mode=[{reduction_mode}] score_percentiles=[{score_percentiles}]")

print("Finished training")

# Save model
# create representative dataset but fake labels as there's no validation dataset for visual-GMM
y = np.zeros(X_train.shape)
representative_data = tf.data.Dataset.from_tensor_slices((X_train, y))

# using None for reduction_mode which gives spatial scores.
# the other options are 'mean' and 'max', we calculate these in Studio classifier using the spatial scores
visual_ad.save_model(dir_path, saved_model_dir, representative_data, reduction_mode=None)