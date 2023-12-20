import sys, os, json

# Hide noisy TF logs (such as CUDA warnings when running on CPU)
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

sys.path.append('./resources/libraries')
from ei_sklearn.gmm_anomaly_detection import GaussianMixtureAnomalyScorer

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filename = "model.tflite"
input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(os.path.join(dir_path, 'options.json')) as f:
    options = json.load(f)

GaussianMixtureAnomalyScorer.run_classify_job(dir_path, model_filename,
    input_filename, options['axes'], output_filename)