from collections import namedtuple

import numpy as np
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.svm import LinearSVC as SVM

coefficient = namedtuple("COEFFICIENT", "lambda1 lambda2 gamma")
EPS = 1e-2


class SMLFDL:
    def __init__(self, K, max_iteration=1):
        self.SVMs = None
        self.D = None
        self.Z = None
        self.K = K  # TODO: rename
        self.classes_ = None
        self.history = []
        self.max_iteration = max_iteration
        self.coe = coefficient(2e-3, 2e-1, 2e-4)
        self.dictionary_learning = DictionaryLearning(
            n_components=self.K, max_iter=5, verbose=True, transform_max_iter=10
        )
        self.SVMs = SVM(multi_class="ovr", dual=False)

    def fit(self, X, y):
        self.X = X  # TODO: better way
        self.y = y  # TODO: what is y
        self.classes_ = np.unique(y)
        self.D = self._initialize_d(X)
        self.Z = self._initialize_z(X)
        epoch = 0
        while not self._converge(epoch) and epoch < self.max_iteration:
            self.SVMs.fit(self.Z.T, self.y.T)

            for i, xi in enumerate(X.T):
                Z_except_i = np.delete(self.Z, i, axis=1)
                y_except_i = np.delete(y, i, axis=0)
                m_yi = np.mean(Z_except_i[:, y_except_i == self.y[i]], axis=1)
                m = np.mean(Z_except_i, axis=1)
                Uzi = self._Uz(i)
                if not Uzi:
                    self.Z[:, i] = self._p_inv(i) @ (
                        self.D.T @ xi.T + self.coe.lambda1 * (2 * m_yi - m)
                    )
                else:
                    # TODO rename
                    svm_coef = self.SVMs.coef_ - self.SVMs.coef_[self.y[i]]
                    svm_b = self.SVMs.intercept_ - self.SVMs.intercept_[self.y[i]] + EPS
                    svm_sum = np.sum(svm_coef * svm_b.reshape(-1, 1), axis=0)
                    self.Z[:, i] = self._q_inv(i) @ (
                        self.D.T @ xi.T + (2 * m_yi - m) - self.coe.gamma * svm_sum
                    )

            # usage of current Z is not clear
            self.D = self.dictionary_learning.fit_transform(X @ self.Z.T)
            epoch += 1

        return self

    def predict(self, X):
        base = self.D.T @ self.D
        base += self.coe.lambda2 * np.eye(base.shape[0])
        X = np.linalg.inv(base) @ self.D.T @ X
        return self.SVMs.predict(X.T)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def _initialize_d(self, X: np.ndarray):
        d = []
        # TODO: add generalization for non aligned sample
        n_components = self.K / len(self.classes_)
        assert (
            self.K % len(self.classes_) == 0
        ), f"SAME ATOMS FEATURE COUNT {self.K}/{len(self.classes_)}"
        for c in range(len(self.classes_)):
            Xc = X[:, self.y == c].T
            assert (
                np.min(Xc.shape) >= n_components
            ), f"PCA COMPONENT ERROR [svd_solver=full] min(X[y=={c}].shape) >= {self.K}/{len(self.classes_)}"
            d.extend(PCA(n_components=int(n_components)).fit(Xc).components_)
        return np.array(d).T

    def _initialize_z(self, X: np.ndarray):
        Z = [
            self._p_inv(i) @ (self.D.T @ xi.T + self.coe.lambda1 * 2)
            for i, xi in enumerate(X.T)
        ]
        return np.array(Z).T

    def _get_base_pq(self, i: int):
        """ :Use once: D.T@D + lambda2*I + lambda1 * (1 - n/n) is same for all computations """
        base = self.D.T @ self.D
        base += self.coe.lambda2 * np.eye(base.shape[0])
        # TODO: self.yi and self.Xi must be removed
        base += self.coe.lambda1 * (
            1 - 2 / np.sum(self.y == self.y[i]) + 1 / self.X.shape[1]
        )
        return base

    def _p_inv(self, i: int):
        p = self._get_base_pq(i)
        return np.linalg.inv(p)

    def _q_inv(self, i: int):
        # TODO: refactor with optimized np.cov
        q = self._get_base_pq(i)
        yi = self.y[i]
        for c in self._Uz(i):
            #  pseudo mal
            wc_wy = self.SVMs.coef_[c] - self.SVMs.coef_[yi]
            q += wc_wy @ wc_wy.T
        return np.linalg.inv(q)

    def _Uz(self, i: int):
        zi = self.Z[:, i]
        yi = self.y[i]
        # TODO: it must not be done separately
        predict_prob_yi = zi.dot(self.SVMs.coef_[yi]) + self.SVMs.intercept_[yi]
        Uzi = [
            c
            for c in range(len(self.classes_))
            if c != yi
            and zi.dot(self.SVMs.coef_[c]) + self.SVMs.intercept_[c] > predict_prob_yi
        ]
        return Uzi

    def _converge(self, epoch):
        return False and self.history


if __name__ == "__main__":
    X_train, y_train = np.load("./data/X.npy"), np.load("./data/y.npy")
    clf = SMLFDL(K=150, max_iteration=3)
    clf.fit(X_train, y_train)
    print("score train", clf.score(X_train, y_train))
    print("finished")

"""
dataset: [scene 15] https://www.kaggle.com/zaiyankhan/15scene-dataset
(250 × 300) varies from 210 to 410

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

image descriptor:
https://github.com/danielshaving/CV_Features_HoG_Feature_Extraction/blob/master/Hog%20Feature%20by%20pure%20numpy.py
"""
