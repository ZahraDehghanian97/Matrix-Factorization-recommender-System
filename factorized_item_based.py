from math import sqrt
from utilities import test,user_feature, book_feature, utility_matrix
import numpy as np


# calculates cosine similarity of two given vectors
def cosine_similarity(person1_dataset, person2_dataset):
    nonzero1 = np.nonzero(person1_dataset)[0]
    nonzero2 = np.nonzero(person2_dataset)[0]
    if len(nonzero2) == 0 or len(nonzero1) == 0: return 0
    dot_product = []
    person_1_sum_square = sum([pow(person1_dataset[item], 2) for item in nonzero1])
    person_2_sum_square = sum([pow(person2_dataset[item], 2) for item in nonzero2])
    for i in list(set(nonzero2) & set(nonzero1)):
        dot_product.append(person1_dataset[i] * person2_dataset[i])
    dot_product = sum(dot_product)
    result = dot_product / (sqrt(person_1_sum_square) * sqrt(person_2_sum_square))
    return result


def find_average_user(test_user):
    item_test_user = np.zeros([len(user_feature[0])])
    sum = 0
    for i in range(number_item):
        if utility_matrix[test_user, i] is not None:
            item_test_user += book_feature[i]*utility_matrix[test_user,i]
            sum += utility_matrix[test_user,i]
    item_test_user/= sum
    return item_test_user


def most_similar_book(average_book, sim_type):
    scores = []
    if sim_type == 'cosine':
        for i in range(number_item):
            scores.append([cosine_similarity(average_book,book_feature[i]), i])
    return sorted(scores, key=lambda t: t[0], reverse=True)[1:]


def item_based(similarity_type):
    for i in range(len(test)):
        test_user = test[i]
        print("\n\nRecommended book for user number = " + str(test_user) + " with " + str(
            similarity_type) + " similarity : ")
        print("------------------------")
        average_book = find_average_user(test_user)
        score_similarity_book = most_similar_book(average_book, similarity_type)
        counter_book = 0
        for j in range( number_item ):
            if not (score_similarity_book[j][1] in np.nonzero(utility_matrix[test_user])[0]):
                counter_book += 1
                print(str(score_similarity_book[j][1]) + " with similarity score = " + str(
                    score_similarity_book[j][0]))
            if counter_book == 5:
                break


print("load pickle file finished")
utility_matrix = utility_matrix
number_user = len(utility_matrix)
number_item = len(utility_matrix[0])
print("number user = " + str(number_user ))
print("number item = " + str(number_item ))
similarities = ['cosine']
# for s in similarities: user_based(s)
for s in similarities: item_based(s)
