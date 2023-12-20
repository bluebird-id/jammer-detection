import jax
import numpy as np
import jax.numpy as jnp
import sklearn
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def translate_function(model, function):
    """Translate a prefit sklearn model & function to a pure jax function.

    Args:
        model: the fit sklearn model to translate.
        function: the specific inference function to translate.
    Returns:
        A non jitted jax version of the function exposing only the x parameter.
    Throws:
        Exception if model has not been .fit()
        Exception if model type has an unsupported configuration.
        Exception if model type or function is not yet supported.
    """

    if sklearn.__version__ != '1.1.1':
        import sys
        print("WARNING: these conversion functions were developed with"
              " sklearn 1.1.1 and may not be correct with"
              f" version {sklearn.__version__}", file=sys.stderr)

    if function == GaussianRandomProjection.transform:
        # first check to ensure .fit has been called
        if not hasattr(model, 'components_'):
            raise Exception("Can't transform model that has not been .fit")

        # bake parameters into standalone function
        def project(x):
            projection_matrix = model.components_
            return jnp.dot(x, projection_matrix.T)
        return project

    elif function == GaussianMixture.score_samples:
        # first check to ensure .fit has been called
        if not hasattr(model, 'means_'):
            raise Exception("Can't transform model that has not been .fit")

        # check covariance type is 'full' (we've only ported covar=full for now)
        if model.covariance_type != 'full':
            raise Exception("Only GMM with covariance_type='full' is supported.")

        # bake parameters into a standalone function for a single element. we
        # do this scalar version since it's easier to port to jax.
        def score_one(x):
            return _GMM_score_single_sample(
                x, model.means_,
                model.weights_, model.precisions_cholesky_)

        # but return vectorised form (since the original score_samples is
        # vectorised)
        score_many = jax.vmap(score_one)
        return score_many

    elif function == StandardScaler.transform:
        # first check to ensure .fit has been called
        if not hasattr(model, 'mean_'):
            raise Exception("Can't transform model that has not been .fit")

        # bake parameters into standalone function
        def standardise(x):
            return (jnp.array(x) - model.mean_) / model.scale_
        return standardise

    else:
        raise Exception(f"unknown function [{str(function)}]")



def _logsumexp(a):
    # numerical stable log sum exp; i.e max -> 0 then added back at end
    # minimal version of https://github.com/scipy/scipy/blob/2e5883ef7af4f5ed4a5b80a1759a45e43163bf3f/scipy/special/_logsumexp.py#L7-L127
    # without the scaling option 'b'
    a_max = jnp.amax(a, axis=0, keepdims=True)
    tmp = a - a_max
    out = jnp.log(jnp.sum(jnp.exp(tmp), axis=0, keepdims=False))
    out += jnp.squeeze(a_max, axis=0)
    return out

def _compute_log_det_cholesky(matrix_chol, n_features):
    # https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/mixture/_gaussian_mixture.py#L354
    n_components, _, _ = matrix_chol.shape
    return jnp.sum(jnp.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1)

def _GMM_score_single_sample(x, means, weights, precisions_cholesky):
    # see https://colab.research.google.com/drive/1OZ0Es53ke0vX4U4rqybD0NAEgCXnWCgp
    # for incremental port from sklearn.mixture._gaussian_mixture._estimate_log_gaussian_prob
    # https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/mixture/_gaussian_mixture.py#L394
    n_components, n_features = means.shape
    log_weights = jnp.log(weights)
    log_det = _compute_log_det_cholesky(precisions_cholesky, n_features)
    # calc seperate components of y separately
    x_prec_chols = jnp.einsum('j,ijk->ik', x, precisions_cholesky)
    mu_prec_chols = jnp.einsum('ij,ijk->ik', means, precisions_cholesky)
    # combine components, then square and sum
    ys = x_prec_chols - mu_prec_chols
    log_prob = jnp.sum(jnp.square(ys), axis=-1)
    estimated_log_prob = -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    # add weighting
    estimated_log_prob += log_weights
    # score via log sum exp
    return _logsumexp(estimated_log_prob)
