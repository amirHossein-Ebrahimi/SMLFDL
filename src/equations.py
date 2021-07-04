# Created by Amir.H Ebrahimi at 2/21/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def D_init(X, y, K, classes):
    n_components = K / len(classes)
    assert K % len(classes) == 0, f"SAME ATOMS FEATURE COUNT {K}/{len(classes)}"

    D = None
    for c in tqdm(range(len(classes)), desc="D-init"):
        Xc = X[:, y == c].T
        assert (
            np.min(Xc.shape) >= n_components
        ), f"PCA COMPONENT ERROR [svd_solver=full] min(X[y=={c}].shape) >= {K}/{len(classes)}"
        pca_components = PCA(n_components=int(n_components)).fit(Xc).components_
        if D is None:
            D = pca_components
        else:
            D = np.vstack((D, pca_components))
    return np.array(D).T
