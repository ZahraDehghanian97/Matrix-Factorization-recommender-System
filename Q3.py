import numpy as np
import pandas as pd
K, N, M = 2, 12, 30
Y_gen = np.random.rand(M, K)
X_1 = np.random.rand(K,int( N/2))
# So that atleast twice as much
X_2 = 2* X_1 + np.random.rand(K,int( N/2))
X_gen = np.hstack([X_2, X_1])
# Normalizing
X_gen = X_gen/np.max(X_gen)
# Creating A (ratings matrix of size M, N)
A = np.dot(Y_gen, X_gen)


def nmf_features(A, k, MAX_ITERS=30, input_constraints_X=None, input_constraints_Y=None):
    import cvxpy as cvx
    np.random.seed(0)

    # Generate random data matrix A.
    m, n = A.shape
    mask = ~np.isnan(A)

    # Initialize Y randomly.
    Y_init = np.random.rand(m, k)
    Y = Y_init

    # Perform alternating minimization.

    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1 + MAX_ITERS):

        # For odd iterations, treat Y constant, optimize over X.
        if iter_num % 2 == 1:
            X = cvx.Variable((k,n))

        # For even iterations, treat X constant, optimize over Y.
        else:
            Y = cvx.Variable((m, k))


        Temp = Y * X
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

X, Y, r = nmf_features(A, 1, MAX_ITERS=20)
print(X)

# import pickle
#
#
# def loadFileFromPickle(fileName):
#     with open(fileName, 'rb') as f:
#         x = pickle.load(f)
#     return x
#
# utility_matrix = loadFileFromPickle('utility.pkl')
# X, Y, r = nmf_features(utility_matrix, 5, MAX_ITERS=20)