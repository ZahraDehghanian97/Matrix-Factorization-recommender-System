from math import sqrt
from numpy import average
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


# calculates pearson similarity of two given vectors
def pearson_similarity(person1_dataset, person2_dataset):
    person1_preferences_sum = average(person1_dataset)
    person2_preferences_sum = average(person2_dataset)
    nonzero1 = np.nonzero(person1_dataset)[0]
    nonzero2 = np.nonzero(person2_dataset)[0]
    if len(nonzero2) == 0 or len(nonzero1) == 0: return 0
    person1_square_preferences_sum = sum(
        [pow((person1_dataset[item] - person1_preferences_sum), 2) for item in nonzero1])
    person2_square_preferences_sum = sum(
        [pow((person2_dataset[item] - person2_preferences_sum), 2) for item in nonzero2])
    product_sum_of_both_users = sum(
        [(person1_dataset[item] - person1_preferences_sum) * (person2_dataset[item] - person2_preferences_sum) for item
         in list(set(nonzero2) & set(nonzero1))])
    similarity = product_sum_of_both_users / sqrt(person2_square_preferences_sum * person1_square_preferences_sum)
    return similarity


def find_average(test_user):
    item_test_user = np.zeros([number_user])
    for i in range(number_item):
        item_test_user += utility_matrix[:,i]*utility_matrix[test_user,i]
    item_test_user/= sum(utility_matrix[test_user])
    return item_test_user


def most_similar_book(average_book, sim_type):
    scores = []
    if sim_type == 'pearson':
        for i in range(number_item):
            scores.append([pearson_similarity(average_book, utility_matrix[:,i]), i])
    elif sim_type == 'cosine':
        for i in range(number_item):
            scores.append([cosine_similarity(average_book,utility_matrix[ i]), i])
    return sorted(scores, key=lambda t: t[0], reverse=True)[1:]


def item_based(similarity_type):
    for i in range(len(test)):
        test_user = test[i]
        print("\n\nRecommended book for user number = " + str(test_user) + " with " + str(
            similarity_type) + " similarity : ")
        print("------------------------")
        average_book = find_average(test_user)
        score_similarity_book = most_similar_book(average_book, similarity_type)
        counter_book = 0
        for j in range(0, number_item - 1):
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
print("number user = " + str(number_user - 1))
print("number item = " + str(number_item - 1))
similarities = ['cosine', 'pearson']
# for s in similarities: user_based(s)
for s in similarities: item_based(s)
