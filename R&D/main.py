# Created by Amir.H Ebrahimi at 1/26/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import time
import numpy as np
from sklearn.svm import LinearSVC
from tqdm import tqdm
from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.decomposition import SparsePCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from collections import namedtuple, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import DictionaryLearning, PCA

np.random.seed(seed=13)
regularization = namedtuple("REGULARIZATION", "e e1 g")
EPS = 1e-2

class SMLFDL:
    """
    X   mxn :m=features :n=data count             - training data
    y   1xn :n=count                              - training label class
    D   mxK :m=features :K=atoms count            - Dictionary K is k for each class
        - K % |classes| = 0                       - Dictionary components= K // |classes|
        - K / |classes| <= min(Xc.shape)          - [PCA_BOUND] for all subset Xc[y == c]
        - m < K << n                              - from two above?
    Z   Kxn :K=atoms count :n=data count          - Used in loss function?
    P   KxK :K=atoms count
    Q   KxK :K=atoms count
    Uz  1*c :c=number of more probable classes
        - c < |classes|

    coe = regularization(λ, λ1, ɣ)
    """

    def __init__(self, K, reg, max_iteration=1):
        self.K = K  # number of atoms
        self.D = None  # dictionary matrix
        self.Z = None  # coefficients matrix
        self.reg = reg  # regularization {lambda lambda1 gamma}
        self.classes_ = None  # unique y classes
        self.history = defaultdict(list)  # history used for convergence test
        self.max_iteration = max_iteration  # maximum iteration count
        self.epoch = None
        # Learning algorithms
        self.dictionary_learning = None
        self._update_dictionary_learning()
        # https://stackoverflow.com/a/23063481/10321531
        self.SVMs = SGDClassifier(loss="hinge", penalty="l2", warm_start=True, n_jobs=2)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.D is None:
            t0 = time.time()
            self.D = self._initialize_d(X, y)
            self.history["D"].append(time.time() - t0)
        if self.Z is None:
            t0 = time.time()
            self.Z = self._initialize_z(X, y)
            self.history["Z"].append(time.time() - t0)
        if self.epoch is None:
            self.epoch = 0

        N = X.shape[1]
        while self.epoch < self.max_iteration:
            print(f"epoch = {self.epoch}")
            t0 = time.time()
            self.SVMs.fit(self.Z.T, y.T)
            self.history["acc"].append(self.score(X, y))
            self.history["SVM"].append(time.time() - t0)
            print('=== SVM ===', self.history['acc'][-1] * 100)
            # D.T@D + e1*I + e (1 + 1/N)
            base = self._d_dagger() + self.reg.e * (1 + 1 / N)

            t0 = time.time()
            for i in tqdm(range(N), desc="update Z"):
                Z_except_i = np.delete(self.Z, i, axis=1)
                y_except_i = np.delete(y, i, axis=0)
                m_yi = np.mean(Z_except_i[:, y_except_i == y[i]], axis=1)
                m = np.mean(Z_except_i, axis=1)
                U_zi = self._probable_classes(y, i)
                update_mat = self._get_inv_matrix(base, y, i, U_zi)
                update_x = self.D.T @ X[:, i] + self.reg.e * (2 * m_yi - m)
                if U_zi:
                    svm_regularization = self.SVMs.coef_[U_zi] - self.SVMs.coef_[y[i]]
                    svm_b = (
                        self.SVMs.intercept_[U_zi] - self.SVMs.intercept_[y[i]] + EPS
                    )
                    svm_sum = np.sum(svm_regularization * svm_b.reshape(-1, 1), axis=0)
                    update_x -= self.reg.g * svm_sum
                self.Z[:, i] = update_mat @ update_x
            self.history['Z'].append(time.time() - t0)

            t0 = time.time()
            self._update_dictionary_learning()
            self.dictionary_learning.fit(X.T)
            self.D = self.dictionary_learning.components_.T
            self.history['D'].append(time.time() - t0)
            self.epoch += 1

        return self

    def predict(self, X):
        X = np.linalg.inv(self._d_dagger()) @ self.D.T @ X
        return self.SVMs.predict(X.T)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


    def save_check_point(self, name: int):
        np.savez_compressed(
            f"./checkpoint/{name}.npz", D=self.D, Z=self.Z, epoch=self.epoch,
        )

    def load_from_checkpoint(self, name: str, y: np.ndarray = None):
        mat = np.load(f"./checkpoint/{name}.npz")
        self.D = mat["D"]
        self.Z = mat["Z"]
        self.epoch = mat["epoch"]
        if y is not None:
            self.SVMs.fit(self.Z.T, y.T)

    def _initialize_d(self, X: np.ndarray, y: np.ndarray):
        n_components = self.K / len(self.classes_)
        assert (
            self.K % len(self.classes_) == 0
        ), f"SAME ATOMS FEATURE COUNT {self.K}/{len(self.classes_)}"

        D = None
        for c in tqdm(range(len(self.classes_)), desc="D-init"):
            Xc = X[:, y == c].T
            assert (
                np.min(Xc.shape) >= n_components
            ), f"PCA COMPONENT ERROR [svd_solver=full] min(X[y=={c}].shape) >= {self.K}/{len(self.classes_)}"
            pca_components = PCA(n_components=int(n_components)).fit(Xc).components_
            if D is None:
                D = pca_components
            else:
                D = np.vstack((D, pca_components))
        return np.array(D).T

    def _initialize_z(self, X: np.ndarray, y: np.ndarray):
        base = self._d_dagger() + self.reg.e * (1 + 1 / X.shape[1])
        N = X.shape[1]
        Z = np.empty((self.K, N))
        for i in tqdm(range(N), desc="Z-init"):
            Z[:, i] = self._get_inv_matrix(base, y, i) @ (
                self.D.T @ X[:, i] + self.reg.e * 2
            )
        return Z

    def _get_inv_matrix(
        self, base: np.ndarray, y: np.ndarray, i: int, U_zi: list = None
    ):
        yi = y[i]
        P = base - self.reg.e * 2 / np.sum(y == yi)
        if not U_zi:  # empty or None
            return np.linalg.inv(P)

        Q = np.power(self.SVMs.coef_[U_zi] - self.SVMs.coef_[yi], 2)
        Q = P + self.reg.g * np.sum(Q)
        return np.linalg.inv(Q)

    def _d_dagger(self):
        base = self.D.T @ self.D + self.reg.e1 * np.eye(self.K)
        return base

    def _svm_decision_function(self, x: np.ndarray, i: int):
        return x.dot(self.SVMs.coef_[i]) + self.SVMs.intercept_[i]

    def _probable_classes(self, y: np.ndarray, i: int):
        zi = self.Z[:, i]
        yi = y[i]
        predict_prob_yi = self._svm_decision_function(zi, yi) + EPS
        U_zi = [
            c
            for c in range(len(self.classes_))
            if c != yi and self._svm_decision_function(zi, c) > predict_prob_yi
        ]
        return U_zi

    def _update_dictionary_learning(self):
        self.dictionary_learning = DictionaryLearning(
            n_jobs=2,
            max_iter=50,
            verbose=True,
            n_components=self.K,
            transform_max_iter=10,
            transform_algorithm="lasso_lars",
            dict_init=self.D.T if self.D is not None else None,
            code_init=self.Z.T if self.Z is not None else None,
        )


if __name__ == "__main__":

    scaler = StandardScaler()

    data = np.load('./data/train-100.npz', allow_pickle=True)
    X_train, y_train, classnames = data['X'], data['y'], data['classnames']
    X_train = scaler.fit_transform(X_train)  # must be in correct format
    X_train = X_train.T
    print(f"{X_train.shape=} {y_train.shape=}")

    clf = SMLFDL(K=15 * 50, reg=regularization(2e-3, 2e-1, 2e-4), max_iteration=1)
    # t0 = time.time()
    clf.fit(X_train, y_train)
    # print("total training time:", time.time() - t0)
    pprint({ k: np.mean(v) for k, v in clf.history.items() })
    # clf.save_check_point("smlfdl.e4.mlp")


    data = np.load('./data/test-10.npz')
    X_test, y_test = data['X'], data['y']
    X_test = scaler.transform(X_test)  # must be in correct format
    X_test = X_test.T
    print(f"{X_test.shape=} {y_test.shape=}")
    # clf.load_from_checkpoint("smlfdl.e4.mlp", y=y_train)
    print("score train", clf.score(X_train, y_train))
    print("score test", clf.score(X_test, y_test))

    print(
        "Linear SVC ",
        LinearSVC(multi_class="ovr")
        .fit(X_train.T, y_train.T)
        .score(X_test.T, y_test.T)
        * 100,
    )


"""
Dictionary learning can also use
# self.dictionary_learning = SparsePCA(
# n_components=self.K,
# V_init=self.D.T if self.D is not None else None,
# U_init=self.Z.T if self.Z is not None else None,
# )
"""


# links
"""
https://www.csie.ntu.edu.tw/~cjlin/libsvm/

Label Consistent K-SVD: recently proposed LLC algorithm
http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html.
"""