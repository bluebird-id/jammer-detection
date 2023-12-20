import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
import numpy as np
from typing import Callable

def construct_weighted_xent_fn(model_output_shape: TensorShape,
                               object_weight: float) -> Callable:
    """ Construct a custom weighted cross entropy function for model.

    Args:
        model_output_shape: output shape of keras model, used for masks etc
        object_weight: loss weight for non background classes (with background
            class weight implied as 1.0)

    Returns:
        loss function suitable for use with keras model.

    When we come to calculate losses we want to use a weighting of 1.0 for the
    background class and use object_weight for all other classes. Tried to do
    this by this reweighting in the loss function but the indexed assignment was
    troublesome with eager tensors. so instead we calculate two masks
    corresponding to the model output; one for the background class, and one for
    all other objects, and sum the losses together.
    """

    if len(model_output_shape) !=4 :
        raise Exception("expected model_output_shape of form (BATCH_SIZE, H, W, NUM_CLASSES")

    _batch_size, height, width, num_classes = model_output_shape
    background_y_true = np.zeros((num_classes,))
    background_y_true[0] = 1.0
    background_loss_mask = np.tile(background_y_true, (height, width, 1))

    # note: the background_loss_mask does not include a batch dimension which
    # is handled during the weighted_xent by broadcasting

    def weighted_xent(y_true, y_pred_logits):
        # run calculation of loss w.r.t weight of 1.0 then use mask to pick out
        # just background values.
        background_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred_logits, pos_weight=1.0)
        background_loss *= background_loss_mask

        # rerun calculation of loss w.r.t weight of pos.weight then use mask to
        # pick out just non background values.
        non_background_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred_logits, pos_weight=object_weight)
        non_background_loss *= 1.0 - background_loss_mask

        # overall loss is just the sum, averaged over all spatial dimensions
        losses = background_loss + non_background_loss
        return tf.reduce_mean(losses)

    return weighted_xent
