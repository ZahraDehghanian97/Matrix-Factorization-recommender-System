import numpy as np
import pickle


def nmf_features(A, k, MAX_ITERS=30):
    import cvxpy as cvx
    np.random.seed(0)
    m, n = A.shape
    mask = ~np.isnan(A)
    Y_init = np.random.rand(m, k)
    Y = Y_init
    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1 + MAX_ITERS):
        if iter_num % 2 == 1:
            X = cvx.Variable((k, n))
        else:
            Y = cvx.Variable((m, k))
        error = A[mask] - (Y * X)[mask]
        obj = cvx.Minimize(cvx.norm(error))
        prob = cvx.Problem(obj)
        prob.solve(solver=cvx.SCS)
        if prob.status != cvx.OPTIMAL:
            pass
        residual[iter_num - 1] = prob.value
        if iter_num % 2 == 1:
            X = X.value
        else:
            Y = Y.value
    return X, Y, residual


def loadFileFromPickle(fileName):
    with open(fileName, 'rb') as f:
        x = pickle.load(f)
    return x


utility_matrix = loadFileFromPickle('utility.pkl')
X, Y, r = nmf_features(utility_matrix, 5, MAX_ITERS=20)
