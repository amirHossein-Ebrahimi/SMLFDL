# Created by Amir.H Ebrahimi at 2/20/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import scipy.io
import numpy as np
from src.utils import reset_random_seed, shrink_k_coded_labels, shuffle_signal_dataset


def spatial_pyramid_features_scene15(
    n_sample_per_category: int = None, seed: int = None
):
    mat = scipy.io.loadmat("../data/spatialpyramidfeatures4scene15.mat")

    data_features = mat["featureMat"]  # 3000x4485
    data_labels = mat["labelMat"]  # 15x4485

    if n_sample_per_category is None or n_sample_per_category <= 0:
        return data_features, data_labels

    reset_random_seed(seed)
    sampled_features = []
    sampled_labels = []
    for class_label_index in range(data_labels.shape[0]):
        current_class_data_features = data_features[
            :, data_labels[class_label_index] == 1
        ]
        current_class_data_labels = data_labels[:, data_labels[class_label_index] == 1]
        indices = np.random.choice(
            current_class_data_labels.shape[1], n_sample_per_category, replace=False
        )
        sampled_features.append(current_class_data_features[:, indices])
        sampled_labels.append(current_class_data_labels[:, indices])

    return np.hstack(sampled_features), np.hstack(sampled_labels)


def load_spatial_pyramid_features_scene15(n_sample_per_category=None, seed=None):
    data_features, k_coded_labels = spatial_pyramid_features_scene15(
        n_sample_per_category, seed
    )
    data_features, data_labels = shuffle_signal_dataset(data_features, k_coded_labels)
    data_labels = shrink_k_coded_labels(data_labels)
    return data_features, data_labels


if __name__ == "__main__":
    data, labels = load_spatial_pyramid_features_scene15(n_sample_per_category=100)
    print(data.shape, labels.shape)
