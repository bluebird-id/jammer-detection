import optax
from jax import jit
import jax.numpy as jnp
import numpy as np
from typing import Callable, List
import time
import logging
import json

logging.getLogger("absl").addFilter(
    logging.Filter("Oryx not found"))

import warnings
# remove future warnings from haiku data_structures.py re: a number of
# utils; tree_structure, tree_unflatten, tree_flatten, etc
warnings.filterwarnings(action='ignore', message=".*jax.tree_util.tree_.*")

import tensorflow as tf
from tensorflow.keras.models import Model
from learned_optimization.research.general_lopt import prefab

def ei_log(msg: str):
    print("EI_LOG_LEVEL=info VELO", msg)


def as_params(keras_model: Model):
    """Extract trainable weights into a pytree with jax arrays.

    Extract the weights from a keras model into pytree with jax arrays
    note: these could have just been named v0, v1, v2..., all that
    matters is the order, but retaining names is easier to debug
    """

    var_names_in_order = []
    params = {}
    for variable in keras_model.trainable_variables:
        var_names_in_order.append(variable.name)
        params[variable.name] = jnp.array(variable)
    return var_names_in_order, params


def train_keras_model_with_velo(
    keras_model: Model,
    training_data: tf.data.Dataset,
    validation_data: tf.data.Dataset,
    loss_fn: Callable,
    num_epochs: int,
    callbacks: List[tf.keras.callbacks.Callback] = None,
):
    """Train a keras model with velo.

    VeLO is a learnt optimiser that uses an LSTM to perform the update
    step. It works well with full batch optimisation so setting as large
    a batch as possible is preferred. It requires more compute than a simpler
    optimiser such as SGC or Adam so works best, especially for large models,
    with a GPU.

    In this utility we run a number of epochs where each epoch includes...
    * collecting gradients with keras gradient tape for the specific loss_fn
    * calculate mean gradients across those batchs
    * do parameter update step using the velo optimiser

    Note specifically that the velo update is only run once per epoch, not per batch.

    For more info see
    * https://arxiv.org/abs/2211.09760
    * https://github.com/google/learned_optimization/tree/main

    Args:
        keras_model: the keras model to train
        training_data: a dataset that returns (x, y_true) batches for training.
                       will be (re)batched to batch_size
        validation_data: a dataset that returns (x, y_true) batches for
                         validation metrics.
        loss_fn: keras loss function.
        num_epochs: number of epochs to configue velo to run for.
        callbacks: list of keras callbacks to run each epoch
    """

    # sequential models might not be built yet, in which case we
    # build them by using the training dataset as a source of x shape
    if not keras_model.built:
        for eg_x, _y in training_data:
            break
        ei_log(f"keras model was unbuilt; building with representative"
               f" x shape of {eg_x.shape}")
        keras_model.build(eg_x.shape)

    # do a dummy compilation of the model using SGD. we'll never actually call
    # fit on the model, but keras needs the model compiled for some things;
    # e.g. callbacks and evaluate.
    # TODO: for extra protection we could pass in a dummy optimiser here that
    #       throws an exception if it's ever used. but that feels like overkill
    #       for now.
    keras_model.compile(optimizer="sgd", loss=loss_fn)

    # TODO: how best to support a class_weight arg? we currently support a
    #      a loss_fn for calling in gradient tape but this is a batch call
    #      to could include a mean reduction, which means it's too late
    #      to do per loss weighting. looking at code it seems more keras
    #      loss functions support a sample_weight so the simplest thing to
    #      do here is expect that class_weighting has been passed to the
    #      the loss_fn created for this call, but that doesn't mirror what
    #      keras does with .fit vs .compile

    # disable some logging related the fact we compile but don't fit with keras
    # ( this is removed just before returning results )
    absl_filter = logging.Filter("These functions will not be directly")
    logging.getLogger("absl").addFilter(absl_filter)

    # init velo ( ignore any future warnings )
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        velo_optimiser = prefab.optax_lopt(num_epochs)
        _var_names, params = as_params(keras_model)
        opt_state = velo_optimiser.init(params)

    # compile the velo update step
    def optax_update_step(params, opt_state, grads_map, loss):
        updates, opt_state = velo_optimiser.update(
            grads_map, opt_state, params=params, extra_args={"loss": loss}
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state
    start_t = time.perf_counter()
    optax_update_step = jit(optax_update_step)
    end_t = time.perf_counter()
    ei_log(f"update jit took {(end_t-start_t):0.5f} sec")

    # collect some stats regarding the time spent between keras gradient
    # calculation and the velo update step
    gradient_calc_timings = []
    update_step_timings = []

    # wrap callbacks in a callback list as keras model fit does
    # ( note: epochs is passed to all callbacks via set_params )
    callbacks = tf.keras.callbacks.CallbackList(
        callbacks=[] if callbacks is None else callbacks,
        add_history=True,
        add_progbar=False,
        model=keras_model,
        epochs=num_epochs
    )
    callbacks.on_train_begin()

    # run training loop
    for epoch in range(num_epochs):
        callbacks.on_epoch_begin(epoch)

        # extract latest keras model variables as params
        var_names_in_order, params = as_params(keras_model)
        if epoch == 0:
            param_sizing = {name: tensor.shape for name, tensor in params.items() }
            ei_log(f"model params and sizing {json.dumps(param_sizing)}")

        # we do gradient calculation from batches so we maintain
        # a gradient and loss sum that we can derive mean values from
        num_gradients = 0
        gradient_sums = {v: 0 for v in var_names_in_order}
        train_loss_sum = 0

        # also maintain a timer used to compare batched gradient calculation
        # time vs velo update step
        start_t = time.perf_counter()

        # keep track of number of examples trained; will be used for logging
        num_examples = 0

        for step, (bx, by) in enumerate(training_data):
            callbacks.on_train_batch_begin(step)

            # do the gradient calculation, in keras, for the batch
            with tf.GradientTape() as tape:
                y_pred = keras_model(bx)
                loss = loss_fn(by, y_pred)

                num_examples += len(bx)

                # all loss_fns calculate a loss per instance, but some also
                # do reduction to a scalar ( almost always with a mean). we
                # don't know which we have, but we want to reduce to a scalar,
                # so if the loss value is _not_ already a scalar we can do a
                # reduce_mean to make it one.
                scalar_value = len(loss.shape) == 0
                if not scalar_value:
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, keras_model.trainable_variables)
                grads_map = {
                    vn: jnp.array(g.numpy()) for vn, g in zip(var_names_in_order, grads)
                }

            # accumulate gradients and loss value
            num_gradients += 1
            for k in gradient_sums.keys():
                gradient_sums[k] += grads_map[k]
            train_loss_sum += float(loss)

            callbacks.on_train_batch_end(step)

        # calculate mean gradients and loss ( from sums accumulated during
        # gradient calculation )
        gradient_means = {k: v / num_gradients for k, v in gradient_sums.items()}
        train_loss_mean = train_loss_sum / num_gradients
        if epoch == 0:
            ei_log(f"gradients averaged from {num_examples} examples over {num_gradients} batches")

        # collect some timing stats re: batched gradient calulation
        end_t = time.perf_counter()
        gradient_calc_timings.append(end_t - start_t)

        # make update step using velo
        start_t = time.perf_counter()
        params, opt_state = optax_update_step(
            params, opt_state, gradient_means, train_loss_mean
        )
        end_t = time.perf_counter()
        update_step_timings.append(end_t - start_t)

        # set weights back in model
        # note: we can't use Model.set_weights since the trainable variables
        # may only be a subset of all the variables
        model_weights_by_name = {w.name: w for w in keras_model.weights}
        for name, weight in params.items():
            model_weights_by_name[name].assign(weight)

        # run evaluation using keras.
        # ( recall:this required the model to be compiled )
        validation_loss = keras_model.evaluate(validation_data, verbose=0)

        # call on_epoch_end on all callbacks
        train_loss_mean = float(train_loss_mean)
        validation_loss = float(validation_loss)
        callbacks.on_epoch_end(
            epoch=epoch,
            logs={"loss": train_loss_mean, "val_loss": validation_loss},
        )

        # give some simple progress
        progress = f"Epoch {epoch}: train loss {train_loss_mean:0.5f}" \
                   f" validation loss {validation_loss:0.5f}"
        print(progress)
        ei_log(progress +
               f" gradient_calc_timing {gradient_calc_timings[-1]:0.5f}" +
               f" update_step_timing {update_step_timings[-1]:0.5f}")


    # log summary of timing stats
    # we expect one outlier ( for first pass / compile etc )
    def deciles(a):
        decile_values = np.percentile(a, np.linspace(0, 100, 11))
        return " ".join([f"{v:0.3f}" for v in decile_values])
    ei_log(f"gradient_calc_timings deciles {deciles(gradient_calc_timings)}")
    ei_log(f"update_step_timings   deciles {deciles(update_step_timings)}")

    # remove absl filter
    logging.getLogger("absl").removeFilter(absl_filter)

    # return keras model history, if it was created
    try:
        return keras_model.history
    except AttributeError:
        return None
