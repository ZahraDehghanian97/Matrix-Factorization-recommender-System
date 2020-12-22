from math import sqrt
from numpy import average
from utilities import test,user_feature, book_feature, utility_matrix
import numpy as np


# calculates cosine similarity of two given vectors
def cosine_similarity(person1, person2):
    person1_dataset = utility_matrix[person1]
    person2_dataset = utility_matrix[person2]
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
def pearson_similarity(person1, person2):
    person1_dataset = utility_matrix[person1]
    person2_dataset = utility_matrix[person2]
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


# returns a sorted list of items based on the given similarity function
def most_similar_user(person, sim_type):
    scores = []
    if sim_type == 'pearson':
        for i in range(number_user):
            scores.append([pearson_similarity(person, i), i])
    elif sim_type == 'cosine':
        for i in range(number_user):
            scores.append([cosine_similarity(person, i), i])
    return sorted(scores,key=lambda t: t[0], reverse=True)[1:]


def get_similarity(user, sim_matrix):
    for row in sim_matrix:
        if row[1] == user:
            return float(row[0])
    return 0


def get_similar(sim_matrix):
    result = []
    for item in sim_matrix:
        result.append(item[1])
    return result


# calculates score of an item based on similar users and their similarities
def compute_score_book(similarity_score):
    final_score = []
    sum_similarity = 0
    for item in similarity_score :
        sum_similarity+=item[0]
    for i in range(number_item):
        final_score.append(
            [sum([(utility_matrix[j, i]) * get_similarity(j, similarity_score) for j in
                  get_similar(similarity_score)])/sum_similarity, i])
    return sorted(final_score, key=lambda t: t[0], reverse=True)


def user_based(similarity_type):
    for i in range(len(test)):
        test_user = test[i]
        print("\n\nRecommended book for user number = " + str(test_user) + " with " + str(
            similarity_type) + " similarity : ")
        print("------------------------")
        similarity_user = most_similar_user(test_user, similarity_type)[:25]  # use 25 similar user
        score_similarity_book = compute_score_book(similarity_user)
        counter_book = 0
        for j in range( number_item ):
            if not (score_similarity_book[j][1] in np.nonzero(utility_matrix[test_user])[0]):
                counter_book += 1
                print(str(score_similarity_book[j][1]) + " with similarity score = " + str(
                    score_similarity_book[j][0]))
            if counter_book == 5:
                break


print("load pickle file finished")
number_user = len(utility_matrix)
number_item = len(utility_matrix[0])
print("number user = " + str(number_user ))
print("number item = " + str(number_item))
similarities = ['cosine']
for s in similarities: user_based(s)
