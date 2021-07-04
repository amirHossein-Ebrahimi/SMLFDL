# Created by Amir.H Ebrahimi at 2/21/21
# Copyright © 2021 Carbon, Inc. All rights reserved.
import numpy as np
from scipy import optimize


def n_svm_objective(W, X, y, reg):
    w, b = W[:-1], W = [-1]

    lambda_w_gamma = w * reg.e
    y_pred = X.T * w + b
    active_idx = y_pred * y < 1
    if not active_idx.any():
        return w.T * lambda_w_gamma
    else:
        active_Y = y[active_idx]
        active_E = y_pred[active_idx] - active_Y
        return active_E.T @ active_E + w.T * lambda_w_gamma


def n_svm_grad(W, X, y, reg):
    w, b = W[:-1], W = [-1]

    lambda_w_gamma = w * reg.e
    y_pred = X.T * w + b
    active_idx = y_pred * y < 1
    if not active_idx.any():
        dw = 2 * lambda_w_gamma
        db = 0
    else:
        active_X = X[:active_idx]
        active_Y = y[active_idx]
        active_E = y_pred[active_idx] - active_Y

        dw = 2 * (active_E.T @ active_X).T + 2 * lambda_w_gamma
        db = 2 * np.sum(active_E)

    return [dw, db]


def lbfgs(X, y, reg):
    n_features, _ = X.shape
    starting_point = np.hstack((np.zeros((1, n_features)), 0))
    
    result = optimize.minimize(
        fun=n_svm_objective,
        x0=starting_point,
        jac=n_svm_grad,
        args=(X, y, reg),
        method="L-BFGS-B",
        tol=None,
        callback=None,
    )
    print(result)


def one_vs_all_optimization(X, y, reg, num_labels):
    n_features, n_samples = X.shape
    
    SVM = []
    rows = X.shape[0]
    params = X.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        # minimize the objective function
        fmin = optimize.minimize(fun=n_svm_objective, jac=n_svm_grad, x0=theta, args=(X, y_i), method='TNC')
        all_theta[i-1,:] = fmin.x
    
    return all_theta
