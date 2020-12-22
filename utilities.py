import pickle
import pandas as pd
import numpy as np
from funk_svd import SVD

test = pd.read_csv('test_users.csv', sep=',', header=None).values[0]
book = pd.read_csv('books.csv', sep=',').values
rating = pd.read_csv('ratings.csv', sep=',')


def saveFileToPickle(fileName, object):
    with open(fileName, 'wb') as f:
        pickle.dump(object, f)
    return


def loadFileFromPickle(fileName):
    with open(fileName, 'rb') as f:
        x = pickle.load(f)
    return x


# creates dictionary of users and books
def createMatrix(dataset):
    number_user = max(dataset[:, 1])
    number_item = max(dataset[:, 0])
    utility_matrix = np.empty([number_user, number_item])
    for row in range(0, np.shape(dataset)[0]):
        utility_matrix[dataset[row][1] - 1, dataset[row][0] - 1] = dataset[row][2]
    return utility_matrix


def save_matrix():
    utility_matrix = createMatrix(rating.values)
    saveFileToPickle('utility.pkl', utility_matrix)
    return


def nmf_features(A, k, MAX_ITERS=30):
    import cvxpy as cvx
    np.random.seed(0)
    m, n = A.shape
    mask = ~np.isnan(A)
    Y_init = np.random.rand(m, k)
    Y = Y_init
    residual = np.zeros(MAX_ITERS)
    for iter_num in range(1, 1 + MAX_ITERS):
        print(iter_num)
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


def save_factorized():
    svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100, n_factors=15, min_rating=1, max_rating=5)
    svd.fit(X=rating)
    print("finish computing factorization")
    saveFileToPickle('user.pkl', svd.pu)
    saveFileToPickle('book.pkl', svd.qi)


# save_factorized()
user_feature = loadFileFromPickle('user.pkl')
book_feature = loadFileFromPickle('book.pkl')
utility_matrix = loadFileFromPickle('utility.pkl')
