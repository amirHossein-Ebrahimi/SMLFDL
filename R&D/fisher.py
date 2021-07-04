# Created by Amir.H Ebrahimi at 1/27/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import DictionaryLearning
from numpy import linalg
from sklearn.decomposition import SparsePCA

if __name__ == "__main__":
    # n_components, n_features = 50, 100
    # n_nonzero_coefs = 17
    # X, D, Z = make_sparse_coded_signal(
    #     n_samples=300,
    #     n_components=n_components,
    #     n_features=n_features,
    #     n_nonzero_coefs=n_nonzero_coefs,
    #     random_state=0,
    # )
    # print(f"{X.shape=}\n{D.shape=}\n{Z.shape=}")
    #
    # print("X - D@Z", np.sum(X - D @ Z))
    # print("dk", np.sum(np.power(D, 2), axis=0))
    #
    # print("X - D@Z", np.sum(X - D @ Z))
    # print("f-norm X-D@Z", linalg.norm(X - D @ Z, ord="fro"))
    # # print('D - X@Z.T', np.sum(D - X@Z.T))
    # # print('f-norm D-X@Z.T', linalg.norm(D - X@Z.T, ord='fro'))
    # # print('Z - D.T@X', np.sum(Z - D.T@X))
    # # print('f-norm Z-D.T@X', linalg.norm(Z - D.T@X, ord='fro'))
    # m = float("+inf")
    # # for i in range(1, 20):
    # #     omp = OrthogonalMatchingPursuit(n_nonzero_coefs=i)
    # #     omp.fit(X.T, Z.T)
    # #     D = omp.coef_.T
    # #     # print(f'{D.shape=}\n{omp.coef_.shape=}')
    # #     norm = linalg.norm(X - D @ Z, ord="fro")
    # #     if norm < m:
    # #         m = norm
    # #         print("X - D@Z", np.sum(X - D @ Z))
    # #         print("updated with i", i)
    # #         print("f-norm X-D@Z", norm)
    # #         print("dk", np.sum(np.power(D, 2)))
    # D = X @ linalg.pinv(Z)
    # print("X - D@Z", np.sum(X - D @ Z))
    # print("f-norm X-D@Z", linalg.norm(X - D @ Z, ord="fro"))
    # dict_learner = DictionaryLearning(
    #     n_components=n_components,
    #     transform_algorithm="lasso_lars",
    #     code_init=Z,
    #     dict_init=D,
    # )
    # dict_learner.fit(X)
    # D2 = dict_learner.components_
    # print(f"{D2.shape=}")
    # print("X - D2@Z", np.sum(X - D2 @ Z))
    # print("f-norm X-D2@Z", linalg.norm(X - D2 @ Z, ord="fro"))
    # print("dk", np.sum(np.power(D2, 2)))

    n_samples = 100
    n_components = 15
    n_features = 20
    X, D, code = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=10,
    )
    # X.shape=(20, 100) (n_features, n_samples)
    # D.shape=(20, 15) (n_features, n_components)
    # code.shape=(15, 100) (n_components, n_samples)
    # X-D@Z = Zero
    dict_learner = DictionaryLearning(
        dict_init=D.T,  # (n_features, n_components).T
        code_init=code.T,  # (n_components, n_samples).T
        n_components=n_components,
    )
    X_transformed = dict_learner.fit_transform(X.T)
    D2 = dict_learner.components_.T
    X2_hat = X_transformed @ D2.T
    # print(
    #     f"""
    #     {X_transformed.shape=}=(20, 100) (n_features, n_samples)
    #     {D2.shape=}=(20, 15) (n_features, n_components)
    # """
    # )

    # dict_learner = DictionaryLearning(n_components=n_components)
    # X_transformed = dict_learner.fit_transform(X.T)
    # D3 = dict_learner.components_.T
    # X3_hat = X_transformed @ D3.T
    
    # SPA
    dict_learner = SparsePCA(
        V_init=D.T,  # (n_features, n_components).T
        U_init=code.T,  # (n_components, n_samples).T
        n_components=n_components,
    )
    X_transformed = dict_learner.fit_transform(X.T)
    D3 = dict_learner.components_.T
    X3_hat = X_transformed @ D3.T
    
    
    print(
        f"""
        X-D@Z = {np.sum(X-D@code)}
        f-norm X-D@Z = {linalg.norm(X - D @ code, ord="fro")}
        dk = {np.sum(np.power(D, 2), axis=0)}
        --------------------------------------------------
        X-D2@Z = {np.sum(X-D2@code)}
        f-norm X-D2@Z = {linalg.norm(X - D2 @ code, ord="fro")}
        d2k = {np.sum(np.power(D2, 2), axis=0)}
        reconstruct2={np.mean(np.sum((X2_hat - X.T) ** 2, axis=1) / np.sum(X.T ** 2, axis=1))}
          --------------------------------------------------
        Sparce PCA
        X-D3@Z = {np.sum(X-D3@code)}
        f-norm X-D3@Z = {linalg.norm(X - D3 @ code, ord="fro")}
        d2k = {np.sum(np.power(D3, 2), axis=0)}
        reconstruct3 = {np.mean(np.sum((X3_hat - X.T) ** 2, axis=1) / np.sum(X.T ** 2, axis=1))}
    """
    )

    # TODO: Find an algorithm which can guess a d which has lowest frobunous norm and has one in columns pow 2 sum
    # then update algorithms


""":cvar
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

import numpy as np
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import DictionaryLearning
from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit

n_components, n_features = 512, 100
n_nonzero_coefs = 17


if __name__ == "__main__":
    y, X, w = make_sparse_coded_signal(n_samples=40,
                                       n_components=n_components,
                                       n_features=n_features,
                                       n_nonzero_coefs=n_nonzero_coefs,
                                       random_state=0)
    print(y.shape, X.shape, w.shape)
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
    omp.fit(X, y)
    coef = omp.coef_.T
    idx_r = coef.nonzero()
    print(coef.shape)
    print(np.sum(w - coef))
    
    # X, y = make_classification(
    #     n_samples=4000, n_classes=10, n_features=305, n_informative=15
    # )
    # print(X.shape, y.shape)
    # classes, counts = np.unique(y, return_counts=True)
    # print(classes, counts)
    # pick = 305
    # dic = {i: 0 for i in classes}
    # X2, y2 = X[y == 0][:pick], y[y == 0][:pick]
    # for i in classes:
    #     if i == 0:
    #         continue
    #     X2 = np.vstack((X2, X[y == i][:pick]))
    #     y2 = np.vstack((y2, y[y == i][:pick]))
    #
    # Z = np.hstack((X2, y2.flatten().reshape(-1, 1)))
    # np.random.shuffle(Z)
    # X2 = Z[:, :-1]
    # y2 = Z[:, -1]
    # X2 = X2 + np.random.random(X2.shape)
    # classes, counts = np.unique(y2, return_counts=True)
    # print(classes, counts)
    #
    # np.save('./data/X.npy', X2.T)
    # np.save('./data/y.npy', y2.T.astype(np.int))
    
    # clf = LinearSVC(random_state=0, tol=1e-5, multi_class="ovr").fit(X, y)
    #
    # x = np.random.random((1, 20))
    # print(clf.predict(x))
    # def asn(i):
    #     return x.dot(clf.coef_[i]) + clf.intercept_[i]
    #
    # res = [asn(i) for i in clf.classes_]
    # print(res)
    # print(max(res))
    # print(np.argmax(res))
    #
    # print(clf)
    # X, dictionary, code = make_sparse_coded_signal(
    #     n_samples=315,
    #     n_components=150,
    #     n_features=3000,
    #     n_nonzero_coefs=10,
    #     random_state=42,
    # )
    # dict_learner = DictionaryLearning(
    #     n_components=150, transform_algorithm="lasso_lars", random_state=42,
    # )
    # dict_learner.fit(X)

"""
