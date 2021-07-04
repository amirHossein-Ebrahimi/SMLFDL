# Created by Amir.H Ebrahimi at 1/25/21
# Copyright © 2021 Carbon, Inc. All rights reserved.
import time
import numpy as np
from collections import namedtuple
from sklearn.svm import LinearSVC
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning, PCA
from sklearn.exceptions import NotFittedError

np.random.seed(seed=1948)
coefficient = namedtuple("COEFFICIENT", "e e1 g")
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
    
    coe = coefficient(λ, λ1, ɣ)
    """

    def __init__(self, K, coe, max_iteration=1):
        self.K = K  # number of atoms
        self.D = None  # dictionary matrix
        self.Z = None  # coefficients matrix
        self.coe = coe  # coefficients {lambda lambda1 gamma}
        self.classes_ = None  # unique y classes
        # self.history = []  # history used for convergence test
        self.max_iteration = max_iteration  # maximum iteration count
        # Learning algorithms
        self.dictionary_learning = None
        self._update_dictionary_learning()
        self.SVMs = LinearSVC(multi_class="ovr")
        self.epoch = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.D is None:
            t0 = time.time()
            self.D = self._initialize_d(X, y)
            print('Initialize D=', time.time() - t0)
        if self.Z is None:
            t0 = time.time()
            self.Z = self._initialize_z(X, y)
            print('Initialize Z=', time.time() - t0)
        if self.epoch is None:
            self.epoch = 0
        elif self.epoch >= self.max_iteration:
            self.SVMs.fit(self.Z.T, y.T)

        while self.epoch < self.max_iteration:
            t0 = time.time()
            self.SVMs.fit(self.Z.T, y.T)
            print('Fit SVM=', time.time() - t0)
            base = self._d_dagger() + self.coe.e * (1 + 1 / X.shape[1])

            t0 = time.time()
            for i, xi in enumerate(X.T):
                Z_except_i = np.delete(self.Z, i, axis=1)
                y_except_i = np.delete(y, i, axis=0)
                m_yi = np.mean(Z_except_i[:, y_except_i == y[i]], axis=1)
                m = np.mean(Z_except_i, axis=1)
                Uzi = self._Uz(y, i)
                update_mat = self._get_matrix(base, y, i, Uzi)
                update_x = self.D.T @ xi.T + self.coe.e * (2 * m_yi - m)
                if Uzi:
                    svm_coefficient = self.SVMs.coef_[Uzi] - self.SVMs.coef_[y[i]]
                    svm_b = self.SVMs.intercept_[Uzi] - self.SVMs.intercept_[y[i]] + EPS
                    svm_sum = np.sum(svm_coefficient * svm_b.reshape(-1, 1), axis=0)
                    update_x -= self.coe.g * svm_sum
                self.Z[:, i] = update_mat @ update_x
            print(f"epoch={self.epoch} Updating Z:", time.time() - t0)

            t0 = time.time()
            self._update_dictionary_learning()
            self.D = self.dictionary_learning.fit_transform(X @ self.Z.T)
            print(f"epoch={self.epoch} Updating D:", time.time() - t0)

            self.epoch += 1

        return self

    def predict(self, X):
        X = np.linalg.inv(self._d_dagger()) @ self.D.T @ X
        return self.SVMs.predict(X.T)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _update_dictionary_learning(self):
        self.dictionary_learning = MiniBatchDictionaryLearning(
            n_components=self.K,
            dict_init=self.D,
            n_iter=10,
            transform_max_iter=10,
            n_jobs=1,
        )

    def _initialize_d(self, X: np.ndarray, y: np.ndarray):
        # TODO: add generalization for non aligned sample
        n_components = self.K / len(self.classes_)
        assert (
            self.K % len(self.classes_) == 0
        ), f"SAME ATOMS FEATURE COUNT {self.K}/{len(self.classes_)}"
        d = []
        for c in range(len(self.classes_)):
            Xc = X[:, y == c].T
            assert (
                np.min(Xc.shape) >= n_components
            ), f"PCA COMPONENT ERROR [svd_solver=full] min(X[y=={c}].shape) >= {self.K}/{len(self.classes_)}"
            d.extend(PCA(n_components=int(n_components)).fit(Xc).components_)
        return np.array(d).T

    def _initialize_z(self, X: np.ndarray, y: np.ndarray):
        base = self._d_dagger() + self.coe.e * (1 + 1 / X.shape[1])
        Z = [
            self._get_matrix(base, y, i) @ (self.D.T @ xi.T + self.coe.e * 2)
            for i, xi in enumerate(X.T)
        ]
        return np.array(Z).T

    def _get_matrix(self, base: np.ndarray, y: np.ndarray, i: int, Uzi: list = None):
        yi = y[i]
        P = base - self.coe.e * 2 / np.sum(y == yi)
        if not Uzi:
            return np.linalg.inv(P)
        # Q
        for c in Uzi:
            wc_wy = self.SVMs.coef_[c] - self.SVMs.coef_[yi]
            P += wc_wy @ wc_wy.T
        return np.linalg.inv(P)

    def _d_dagger(self):
        base = self.D.T @ self.D + self.coe.e1 * np.eye(self.K)
        return base

    def _svm_decision_function(self, x: np.ndarray, i: int):
        return x.dot(self.SVMs.coef_[i]) + self.SVMs.intercept_[i]

    def _Uz(self, y: np.ndarray, i: int):
        zi = self.Z[:, i]
        yi = y[i]
        predict_prob_yi = self._svm_decision_function(zi, yi) + EPS
        Uzi = [
            c
            for c in range(len(self.classes_))
            if c != yi and self._svm_decision_function(zi, c) > predict_prob_yi
        ]
        return Uzi

    def save_check_point(self, name: int):
        np.savez_compressed(
            f"./checkpoint/{name}.npz", D=self.D, Z=self.Z, epoch=self.epoch,
        )

    def load_from_checkpoint(self, name: str):
        mat = np.load(f"./checkpoint/{name}.npz")
        self.D = mat["D"]
        self.Z = mat["Z"]
        self.epoch = mat["epoch"]


if __name__ == "__main__":
    X_train, y_train = np.load("./data/X.npy"), np.load("./data/y.npy").astype(np.int)
    print(f"{X_train.shape=} {y_train.shape=}")
    clf = SMLFDL(K=50 * 15, coe=coefficient(2e-3, 2e-1, 2e-4), max_iteration=1)
    t0 = time.time()
    # clf.load_from_checkpoint("e-60")
    clf.fit(X_train, y_train)
    # clf.save_check_point("e-")

    print("total training time:", time.time() - t0)
    print("score train", clf.score(X_train, y_train) * 100)
    print(
        "Linear SVC ",
        LinearSVC(multi_class="ovr")
        .fit(X_train.T, y_train.T)
        .score(X_train.T, y_train.T)
        * 100,
    )

    print("finished")
    # dataset: [scene 15] https://www.kaggle.com/zaiyankhan/15scene-dataset
    # (250 × 300) varies from 210 to 410



"""
pissed off
X_train.shape=(3000, 1500) y_train.shape=(1500,)
Initialize D= 0.5064208507537842
Initialize Z= 36.2189245223999
Fit SVM= 95.79779601097107
sklearn/svm/_base.py:946:
ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
warnings.warn("Liblinear failed to converge, increase")
epoch=0 Updating Z: 42.604549407958984
sklearn/decomposition/_dict_learning.py:174:
RuntimeWarning:  Orthogonal matching pursuit ended prematurely due to linear dependence in the dictionary.
The requested precision might not have been met.
epoch=0 Updating D: 27.29347252845764
total training time: 202.47106885910034
score train 6.800000000000001
Linear SVC  100.0
finished
"""



"""
Dictionary learning can also use
# self.dictionary_learning = SparsePCA(
# n_components=self.K,
# V_init=self.D.T if self.D is not None else None,
# U_init=self.Z.T if self.Z is not None else None,
# )
"""
