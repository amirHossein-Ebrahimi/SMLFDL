# Created by Amir.H Ebrahimi at 2/21/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


DEFAULT_SEED = 1984


def reset_random_seed(seed: int = DEFAULT_SEED):
    if seed is None:
        seed = DEFAULT_SEED
    np.random.seed(seed)
    random.seed(seed)


def shrink_k_coded_labels(k_coded_labels):
    return np.argmax(k_coded_labels, axis=0)


def shuffle_signal_dataset(signal_dataset, signal_k_coded_labels):
    X, y = shuffle(signal_dataset.T, signal_k_coded_labels.T)
    return X.T, y.T


def report_metrics(y_true, y_pred, prefix="SMLFDL"):
    print(f"{prefix}::accuracy= {(accuracy_score(y_true, y_pred) * 100):.3f}%")


def random_signal_classifier(size, n_classes):
    return np.random.choice(n_classes, size)
