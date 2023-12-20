import os
from pathlib import Path
import requests
from typing import Optional

def get_or_download_pretrained_weights(weights_prefix: str, num_channels: int, alpha: float, allowed_combinations: list) -> str:
    # Check if there's a dictionary in allowed_combinations with matching num_channels and alpha
    if not any(combination['num_channels'] == num_channels and combination['alpha'] == alpha for combination in allowed_combinations):
        raise Exception(
            f"Pretrained weights not currently available for num_channel={num_channels} with alpha={alpha}."
            f" Current supported combinations are {allowed_combinations}."
            " For further assistance please contact support at https://forum.edgeimpulse.com/"
        )

    weights_mapping = {
        (1, 0.1): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.grayscale.bsize_64.lr_0_05.epoch_441.val_loss_4.13.val_accuracy_0.2.hdf5",
        (1, 0.35): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_35.96x96.grayscale.bsize_64.lr_0_005.epoch_260.val_loss_3.10.val_accuracy_0.35.hdf5",
        (3, 0.1): "transfer-learning-weights/edgeimpulse/MobileNetV2.0_1.96x96.color.bsize_64.lr_0_05.epoch_498.val_loss_3.85.hdf5",
        (3, 0.35): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96.h5",
        (3, 1.0): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96.h5"
    }

    weights = os.path.join(weights_prefix, weights_mapping[(num_channels, alpha)])

    # Explicit check that requested weights are available.
    if (weights and not os.path.exists(weights)):
        p = Path(weights)
        if not p.exists():
            if not p.parent.exists():
                p.parent.mkdir(parents=True)
            root_url = 'https://cdn.edgeimpulse.com/'
            weights_data = requests.get(root_url + os.path.relpath(weights, weights_prefix)).content
            with open(weights, 'wb') as f:
                f.write(weights_data)

    return weights

def get_weights_path_if_available(weights_prefix: str, num_channels: int, alpha: float, dimension: int) -> Optional[str]:
    weights_mapping = {
        (3, 0.35, 224): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5",
        (3, 0.35, 192): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_192_no_top.h5",
        (3, 0.35, 160): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160_no_top.h5",
        (3, 0.35, 128): "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5",
        (3, 0.35, 96):  "transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96_no_top.h5"
    }

    weights = os.path.join(weights_prefix, weights_mapping[(num_channels, alpha, dimension)])

    # return weights path if exists
    if (weights and os.path.exists(weights)):
        return weights
    else:
        return None