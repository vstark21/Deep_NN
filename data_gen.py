import numpy as np

def data(m, seed=0):
    np.random.seed(seed)
    X = np.random.randn(3, m)
    Y = np.zeros((m))
    for x1, x2, x3, i in zip(X[0], X[1], X[2], range(m)):
        if 3 * x1 - x2 + 2 * x3 + np.random.rand() >= 0:
            Y[i] = 1
    return X, Y 

def cat_data(m, seed=0):
    np.random.seed(seed)
    X = np.random.randn(3, m)
    X = np.round(X, 0)
    Y = np.zeros((2, m))
    for x1, x2, x3, i in zip(X[0], X[1], X[2], range(m)):
        if 3 * x1 - x2 + 2 * x3 >= 0:
            Y[0][i] = 1
        else:
            Y[1][i] = 1
    return X, Y

def one_hot(num_classes, targets):
    Y = np.zeros((num_classes, targets.shape[0]))
    for i, el in enumerate(targets):
        Y[el][i] = 1
    return Y

