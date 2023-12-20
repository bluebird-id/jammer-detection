import sys, os, json
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable annoying warning when GPU not found https://github.com/google/jax/issues/6805
logging.getLogger("absl").addFilter(logging.Filter("No GPU/TPU found, falling back to CPU."))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

sys.path.append('./resources/libraries')
from ei_sklearn.gmm_anomaly_detection import GaussianMixtureAnomalyScorer
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'options.json')) as f:
    options = json.load(f)

model_filename_tflite = 'model.tflite'
model_filename_pickle = 'gmm.pickle'
features_filename = 'X_train_features.npy'
metadata_filename = 'model_metadata.json'
gmm_metadata_filename = 'gmm_metadata.json'

gmm = GaussianMixtureAnomalyScorer(options=options)
gmm.run_training(dir_path, features_filename, options['axes'])
gmm.save_model(dir_path, model_filename_tflite, model_filename_pickle,
    features_filename, metadata_filename, gmm_metadata_filename, options['axes'])

# save scores of training data, to be used in anomaly explorer
output_filename = 'cache-gmm-training-scores.json'
gmm.predict_samples(
    dir_path, features_filename, options['axes'], output_filename)