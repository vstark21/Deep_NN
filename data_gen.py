import numpy as np

def data(m, seed):
    np.random.seed(seed)
    X = np.random.randn(3, m)
    Y = np.zeros((m))
    for x1, x2, x3, i in zip(X[0], X[1], X[2], range(m)):
        if 3 * x1 - x2 + 2 * x3 >= 0:
            Y[i] = 1
    return X, Y 
