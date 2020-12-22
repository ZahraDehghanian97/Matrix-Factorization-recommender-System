import numpy as np
from math import sqrt
from utilities import test, user_feature, book_feature, utility_matrix

expected_utility_matrix = np.matmul(user_feature, book_feature.T)


def find_average(test_user):
    item_test_user = expected_utility_matrix[test_user]
    return item_test_user


def most_similar_book(average_book):
    scores = []
    for i in range(number_item):
        scores.append([average_book[i], i])
    return sorted(scores, key=lambda t: t[0], reverse=True)[1:]


def multiply_based():
    for i in range(len(test)):
        test_user = test[i]
        print("\n\nRecommended book for user number = " + str(test_user))
        print("------------------------")
        user_vector = find_average(test_user)
        score_similarity_book = most_similar_book(user_vector)
        counter_book = 0
        for j in range(0, number_item):
            if not (score_similarity_book[j][1] in np.nonzero(utility_matrix[test_user])[0]):
                counter_book += 1
                print(str(score_similarity_book[j][1]) + " with score = " + str(
                    score_similarity_book[j][0]))
            if counter_book == 5:
                break


print("load pickle file finished")
number_user = len(utility_matrix)
number_item = len(utility_matrix[0])
print("number user = " + str(number_user))
print("number item = " + str(number_item))
multiply_based()
