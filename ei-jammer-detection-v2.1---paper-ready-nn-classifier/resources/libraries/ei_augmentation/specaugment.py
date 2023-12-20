import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import random

class SpecAugment():
    """Implements the SpecAugment spectrogram data augmentation technique outlined in https://arxiv.org/abs/1904.08779"""

    def __init__(self, spectrogram_shape,  mT_num_time_masks=0, T_time_mask_max_consecutive=0,
                    p_time_mask_max_percentage=1., mF_num_freq_masks=0, F_freq_mask_max_consecutive=0,
                    mask_with_mean=True, enable_time_warp=False, W_time_warp_max_distance=1, debug=False):

        self.shape = spectrogram_shape

        self.mT = mT_num_time_masks
        self.T = T_time_mask_max_consecutive
        self.p = p_time_mask_max_percentage
        # This is the maximum number of time columns we are allowed to mask
        self.max_cols_permitted = int(self.p * self.shape[0])

        self.mF = mF_num_freq_masks
        self.F = F_freq_mask_max_consecutive

        self.mask_with_mean = mask_with_mean
        self.mean = None

        self.W_time_warp_max_distance = W_time_warp_max_distance
        self.enable_time_warp = enable_time_warp

        self.debug = debug

    def log(self, *args):
        if self.debug:
            print(*args)

    def time_mask(self, spectrogram):
        self.log('masking times')
        for _ in range(self.mT):
            # Decide how many time bands to mask, but no more than p * number of time buckets
            self.log('max_cols_desired', self.T)
            self.log('max_cols_permitted', self.max_cols_permitted)
            num_cols_to_mask = np.random.randint(0, high=min(self.T, self.max_cols_permitted) + 1)
            self.log('num_cols_to_mask', num_cols_to_mask)
            if num_cols_to_mask == 0:
                continue
            # Choose where the mask should be applied
            mask_start = np.random.randint(0, high=self.shape[0] - num_cols_to_mask)
            self.log('mask_start', mask_start)
            # Chop out the parts we want to keep and replace the rest with zeros
            first_part = spectrogram[0:mask_start, :]
            self.log('first_part shape', first_part.shape)
            if self.mask_with_mean:
                mask_with = self.mean
            else:
                mask_with = 0.
            self.log('masking with', mask_with)
            mask_part = tf.fill((num_cols_to_mask, self.shape[1]), mask_with)
            self.log('mask_part shape', mask_part.shape)
            remaining_cols = self.shape[0] - (mask_start + num_cols_to_mask)
            self.log('remaining_cols', remaining_cols)
            if remaining_cols > 0:
                last_part = spectrogram[-remaining_cols:, :]
                self.log('last_part shape', last_part.shape)
                spectrogram = tf.concat([first_part, mask_part, last_part], 0)
            else:
                spectrogram = tf.concat([first_part, mask_part], 0)

        return spectrogram

    def freq_mask(self, spectrogram):
        self.log('masking frequencies')
        for _ in range(self.mF):
            # Decide how many frequency bands to mask
            self.log('max_rows_desired', self.F)
            num_rows_to_mask = np.random.randint(0, high=self.F + 1)
            self.log('num_rows_to_mask', num_rows_to_mask)
            if num_rows_to_mask == 0:
                continue
            # Choose where the mask should be applied
            mask_start = np.random.randint(0, high=self.shape[1] - num_rows_to_mask)
            self.log('mask_start', mask_start)
            # Chop out the parts we want to keep and replace the rest with zeros
            first_part = spectrogram[:, 0:mask_start]
            self.log('first_part shape', first_part.shape)
            if self.mask_with_mean:
                mask_with = self.mean
            else:
                mask_with = 0.
            self.log('masking with', mask_with)
            mask_part = tf.fill((self.shape[0], num_rows_to_mask), mask_with)
            self.log('mask_part shape', mask_part.shape)
            remaining_rows = self.shape[1] - (mask_start + num_rows_to_mask)
            self.log('remaining_rows', remaining_rows)
            if remaining_rows > 0:
                last_part = spectrogram[:, -remaining_rows:]
                self.log('last_part shape', last_part.shape)
                spectrogram = tf.concat([first_part, mask_part, last_part], 1)
            else:
                spectrogram = tf.concat([first_part, mask_part], 1)

        return spectrogram

    def warp(self, spectrogram):
        self.log('warping')
        # Reshape to [Batch_size, freq, time, 1] for compatibility with sparse_image_warp
        spectrogram = tf.transpose(spectrogram)
        spectrogram = tf.reshape(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1], 1))

        v, tau = spectrogram.shape[1], spectrogram.shape[2]
        horiz_line_thru_ctr = spectrogram[0][v//2]

        # Pick a random point along the time axis
        random_pt = horiz_line_thru_ctr[random.randrange(self.W_time_warp_max_distance, tau - self.W_time_warp_max_distance)]
        # Pick a distance to warp the point
        w = np.random.uniform((-self.W_time_warp_max_distance), self.W_time_warp_max_distance) # distance

        src_points = [[[v//2, random_pt[0]]]]
        dest_points = [[[v//2, random_pt[0] + w]]]

        spectrogram, _ = tfa.image.sparse_image_warp(spectrogram, src_points, dest_points, num_boundary_points=2)

        # Reshape back into a 2D spectrogram
        spectrogram = tf.reshape(spectrogram, (spectrogram.shape[1], spectrogram.shape[2]))
        spectrogram = tf.transpose(spectrogram)

        return spectrogram

    def mapper(self):
        """Returns a function that will distort a given spectrogram and can be passed into dataset.map"""
        def augment(data, label):
            original_shape = data.shape
            # Reshape from a flat structure into a spectrogram
            spectrogram = tf.reshape(data, self.shape)
            # Calculate mean now since it will change as the tensor is transformed
            if self.mask_with_mean:
                self.mean = tf.math.reduce_mean(spectrogram)
            # Apply warping if enabled
            if self.enable_time_warp:
                spectrogram = self.warp(spectrogram)
            # Apply time masking
            if self.mT != 0 and self.T != 0:
                spectrogram = self.time_mask(spectrogram)
            # Apply freq masking
            if self.mF != 0 and self.F != 0:
                spectrogram = self.freq_mask(spectrogram)
            # Restore the original shape
            return tf.reshape(spectrogram, original_shape), label

        return augment


if __name__ == '__main__':

    data = tf.constant([-1.2088, 0.8077, -1.2718, 0.9296, 0.3681, 0.6105, -1.2310, -0.4794, -1.5568, -0.2165, -0.3117, 1.7763, 1.0199, -1.2463, 1.1323, -0.6742, 0.5428, -0.0934, 1.1281, -0.1304, -1.5206, -0.8680, 0.4578, 1.5305, -0.3216, -1.0093, -1.1936, 0.3925, -0.4553, 0.0068, -0.0291, 1.9161, 1.2346, 0.1831, 0.4270, 0.8814, -0.4925, 1.1664, -0.4602, -0.8706, 0.1529, 0.6830, 0.6329, 2.0092, 2.0497, 0.3300, 0.4155, -0.6645, -0.3442, 0.8807, 1.8263, 0.0459, 0.1647, 1.3259, 3.7239, 1.2927, 0.8782, 1.9258, 1.2279, -0.8569, -0.7764, 1.0469, -1.2202, 2.1302, -1.7788, 0.5244, 0.4896, 3.2340, 0.6358, -0.4479, 0.1797, 0.6481, 0.9170, 0.1367, 2.7996, -1.4084, 2.4027, 2.2105, 0.7029, 0.7498, 2.9662, 1.0103, -2.2953, -0.0628, -1.4134, -1.1034, -0.3710, 1.5226, -2.0488, 1.3371, 1.2007, 0.7765, 1.0675, 0.6835, 0.0676, -1.8066, -0.2722, -0.0429, 0.8298, 0.6440, 1.7255, -1.0896, -0.1189, 0.7776, 0.7243, 0.9258, 0.2788, 0.2132, -1.8692, -0.3523, 0.9736, 1.3935, 1.5471, 1.3356, -0.6553, -0.3211, 1.6873, 0.7650, 1.5794, 0.0680, -0.4493, -2.4868, -1.8624, 0.5463, 2.8869, 0.1726, 1.4224, 2.0244, 1.3945, 3.3454, 0.6195, 1.3790, 0.3459, 1.3320, -1.5382, -0.9439, -0.1156, 2.0903, 0.3496, 1.9157, 1.2440, -0.5371, -0.0704, 1.2192, 1.0394, 0.3933, 0.3174, 0.1075, 0.5312, 0.0653, -0.4844, -0.7180, 0.9674, 0.1952, -1.9273, -0.4474, 1.3186, -0.1743, -0.3990, 0.0473, -0.0697, 0.5485, 0.8901, -0.0348, 0.8493, -0.2651, -0.4752, -0.7235, -1.1606, 1.3134, -0.3467, 0.8079, -0.2622, -0.0572, -1.2945, -1.3981, -1.8074, 0.7918, 0.8415, -0.1358, -1.2189, 0.3357, 1.2760, -0.4987, 0.0437, -0.6443, 0.0288, -0.4869, -0.3159, -0.2528, 0.7601, 0.0818, -0.2047, -0.2441, -0.0295, 1.2420, -0.7473, 0.5634, -0.4144, 0.0273, -1.3820, -0.3762, 0.2883, 1.7204, -1.7811, -1.0218, -1.4836, -0.4715, 1.3110, -0.4870, 0.7543, -0.7539, 1.1708, -0.5595, 0.1587, -1.4375, 1.3221, -1.0455, -1.1990, -0.4658, 0.1682, 1.2555, -1.0021, 0.1574, -1.1977, 0.5042, 0.2073, -0.1160, -0.7683, -1.0793, -1.6877, 0.4569, 0.1990, -0.3676, 1.0962, -1.0299, -0.0976, -1.3744, 0.2398, -1.4706, 0.1412, 1.2016, 0.9595, -0.9822, -0.6120, -0.2200, 0.0603, 1.1433, -1.9573, -0.8625, -1.9108, -0.8033, -1.3281, 0.8052, -0.1136, 1.7010, 0.1231, 0.1943, -0.4508, 0.0698, 1.0427, -2.0553, -0.4589, -1.5444, 0.0314, -1.6923, -0.4358, 0.8467, 1.4460, -0.4972, 0.2851, 0.5290, -0.2595, 0.9790, -0.8397, 0.1004, -0.1400, -0.0332, 0.1931, 0.3563, -1.7129, 0.5880, 0.3908, -0.1009, 0.8137, -0.7212, 1.0208, -0.4930, 0.0607, -0.1876, 0.6928, -0.2318, 1.7786, -0.7679, 0.7816, -0.9222, -0.7061, -0.5457, -0.3683, 0.9377, -0.8969, 0.0967, -0.1587, 1.1340, 0.1316, 1.2760, -0.1600, -0.3636, -0.7415, -0.9420, -0.5513, 0.2132, 0.9532, -2.3730, -1.0264, -1.1505, 1.6279, -0.6457, 0.3894, -1.2127, -1.1097, -0.5316, 1.7825, 1.7504, 1.2029, 0.8776, -1.6849, -0.1147, -1.5376, -0.9779, -0.5200, 0.9883, 0.3753, 0.8734, 0.1962, 0.1936, -1.0732, -0.5491, 0.6875, -1.6980, -0.3765, -2.0229, 0.5165, -0.4948, -1.3042, -0.3127, -0.7749, -1.0481, 0.9232, -1.1663, -0.2371, 0.5220, -0.5705, -0.3548, -1.5087, 1.1961, -0.7109, 0.1368, 0.2613, -0.4653, -0.6131, 1.2938, 0.5245, -0.1013, 0.2570, -0.4867, -1.2531, -1.2418, 1.1883, -0.1317, 0.9631, 0.1517, 0.7892, -1.1968, -0.0033, 0.1131, -0.7715, 0.1601, 0.5691, 0.0480, -1.1296, 0.3650, -0.9593, 0.4008, 0.8039, 0.0131, -0.5821, 0.7679, -0.0084, -0.5161, -0.0915, -0.7610, -0.5263, -1.1952, 0.4823, 0.0654, 1.7872, 1.9409, 0.6154, 0.2116, 0.6905, -0.4474, -0.8948, -0.1668, -0.5002, -0.3913, -1.2270, -0.4263, 0.9001, 0.5570, 0.1423, 0.2106, 0.4388, 1.1700, 0.1545, -0.5823, -0.3220, -0.8828, -0.4061, -0.2257, 1.2085, 0.1519, 1.0419, 0.8786, -0.4502, 0.0558, -0.0964, -1.7337, -1.8942, -0.4494, -0.5134, -0.2189, 0.0911, 1.4091, -1.0744, 0.0830, -0.1274, -0.6046, 0.4026, 0.2022, -0.6420, -0.3656, -0.5669, -0.6483, -0.6389, 0.3625, 1.0843, 0.4824, 1.0040, -0.8905, 0.4504, -1.2669, -0.4296, 0.4530, 0.6932, -0.7936, -0.2866, -0.2409, 1.2335, 0.0193, -1.5212, -1.6114, -1.3487, 0.8867, -0.4547, -0.8581, -0.6277, 0.2320, -0.9107, 0.6569, 0.0645, 0.8901, -0.0753, 0.0917, -0.6448, -1.1055, -0.8080, -0.6601, -0.7272, -0.8012, 0.3291, -1.0788, 0.6154, 0.3793, 1.4070, -0.4819, 1.0050, 1.6202, 0.7117, -0.2742, -1.6884, -2.7911, -1.6802, -1.4690, -1.2032, 0.8257, -0.6556, 0.4075, -0.6502, -0.3438, -1.2768, -0.4835, 1.9755, 1.1757, -0.7304, -0.4029, 0.6223, -1.3116, 1.3248, 0.0921, 1.0004, 0.2144, 0.5698, -1.5132, -0.9613, -1.1974, -1.3895, -0.5183, 0.2261, 0.1437, -1.3257, 0.7024, 0.5240, 1.7662, -0.8599, -0.6286, -1.1931, -0.5298, -1.0559, 0.2855, 1.1823, -0.0509, -1.0048, -1.3030, 0.4577, -0.8741, 0.5491, -0.2697, -0.2673, -1.0705, 0.7903, -0.5391, 0.0951, 0.7719, -0.2378, -0.2728, -1.3119, 0.9315, -0.4027, 0.2437, -0.3095, 0.3653, -1.8972, -0.2431, -0.7767, 0.1057, 1.0116, -0.6123, -0.3787, -1.2339, 1.1566, -0.2515, 1.9012, 0.1502, 0.0265, -1.7136, -0.5068, -2.7786, -1.2458, 0.2502, -0.3021, -0.8795, -1.2928, 1.1228, -0.3826, 0.6818, 0.0129, 1.0151, 0.9699, 0.2817, -2.3435, 0.3986, -1.2744, -0.0562, 0.6707, -1.2555, 0.5295, -1.5219, 0.3446, 0.4387, 0.8038, -1.4466, -0.2310, 0.2405, -0.0876, 0.2007, -0.1471, 0.8549, -1.2164, -0.3641, -0.4911, 0.8378, 1.1878, 2.3727, -0.2757, 1.0579, -0.2396, -0.3161, 0.4150, 1.2716, 2.3754, -1.2071, 0.6320, -1.0850, 0.4093, -0.6954, 1.6009, -0.0447, 1.1852, -0.2229, 0.5322, 1.0830, 0.2428, -0.6864, -1.2132, 0.8309, -0.3941, 1.2766, -2.0414, 0.0148, -0.7857, -0.0837, -0.1524, 0.1162, 1.2181, 0.9404, -0.4284])
    print('Input shape is', data.shape)
    shape = [49, 13]

    augment = SpecAugment(shape, mT_num_time_masks=0, T_time_mask_max_consecutive=3,
                    p_time_mask_max_percentage=1., mF_num_freq_masks=0, F_freq_mask_max_consecutive=3,
                    mask_with_mean=True, enable_time_warp=True, W_time_warp_max_distance=6, debug=True)
    map_fn = augment.mapper()

    output, label = map_fn(data, 'yes')

    print('Output shape is', output.shape)

    import matplotlib.pyplot as plot
    import seaborn as sns
    original_two_dee = np.reshape(data, (49, -1))
    sns.heatmap(original_two_dee, linewidth=0.5)
    plot.show()

    transformed_two_dee = np.reshape(output, (49, -1))
    sns.heatmap(transformed_two_dee, linewidth=0.5)
    plot.show()
