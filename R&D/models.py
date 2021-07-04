# Created by Amir.H Ebrahimi at 1/23/21
# Copyright © 2021 Carbon, Inc. All rights reserved.

import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def svm_test():
    svm = LinearSVC(random_state=0)
    X = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]).reshape(-1, 1)
    y = np.array([-1 if i < 1 else 1 for i in X])
    svm.fit(X.reshape(-1, 1), y)
    plt.scatter(X, y, s=80)
    b = svm.intercept_
    w = svm.coef_
    print(f"{X.shape=}")
    print(f"{b.shape=}")
    print(f"{w.shape=}")
    print(f"{svm.classes_=}")
    # plt.plot(X, X @ w + b)
    plt.plot(X, X @ w + b)

    plt.show()
    test = np.array([-5, -1, 0, 1, 3, 19]).reshape(-1, 1)

    print(f"{svm.decision_function(test)=}")
    print(f"{test @ w + b=}")
    print(svm.predict(test))


if __name__ == "__main__":
    svm_test()
