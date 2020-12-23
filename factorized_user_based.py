from math import sqrt
from utilities import test, user_feature, utility_matrix
import numpy as np


# calculates cosine similarity of two given vectors
def cosine_similarity(person1, person2):
    person1_dataset = user_feature[person1]
    person2_dataset = user_feature[person2]
    nonzero1 = np.nonzero(person1_dataset)[0]
    nonzero2 = np.nonzero(person2_dataset)[0]
    if len(nonzero2) == 0 or len(nonzero1) == 0: return 0
    dot_product = []
    person_1_sum_square = sum([pow(person1_dataset[item], 2) for item in nonzero1])
    person_2_sum_square = sum([pow(person2_dataset[item], 2) for item in nonzero2])
    for i in list(set(nonzero2) & set(nonzero1)):
        dot_product.append(person1_dataset[i] * person2_dataset[i])
    dot_product = sum(dot_product)
    # if dot_product< 0 : dot_product*= -1
    result = dot_product / (sqrt(person_1_sum_square) * sqrt(person_2_sum_square))
    return result


# returns a sorted list of items based on the given similarity function
def most_similar_user(person, sim_type):
    scores = []
    sum_similarity = 0
    for i in range(number_user):
        temp = cosine_similarity(person, i)
        sum_similarity+=temp
        scores.append([temp, i])
    for i in range(number_user):
        scores[i][0]=scores[i][0]/sum_similarity
    return sorted(scores, key=lambda t: t[0], reverse=True)[1:]



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
    similars = get_similar(similarity_score)
    # sum_similarity = 0
    # for row in similarity_score :
    #     sum_similarity+= row[0]
    for i in range(number_item):
        sum = 0
        # sum_similarity = 0
        for j in similars:
            if not np.isnan( utility_matrix[j, i]):
                s = get_similarity(j, similarity_score)
                sum += (utility_matrix[j, i] * s)
                # sum_similarity += s
        # if sum_similarity >0 :
        #     sum /= sum_similarity
        final_score.append([sum, i])
    return sorted(final_score, key=lambda t: t[0], reverse=True)


def notNull(row):
    index = []
    for i in range(len(row)):
        if not np.isnan(row[i]) :
            index.append(i)
    return index


def user_based(similarity_type):
    for i in range(len(test)):
        test_user = test[i]
        print("\n\nRecommended book for user number = " + str(test_user) + " with " + str(
            similarity_type) + " similarity : ")
        test_user -=1
        print("------------------------")
        similarity_user = most_similar_user(test_user, similarity_type)[:100]  # use 25 similar user
        score_similarity_book = compute_score_book(similarity_user)
        counter_book = 0
        print("read book : "+str(np.add(notNull(utility_matrix[test_user]),1).tolist()))
        for j in range(number_item):
            if not (score_similarity_book[j][1] in notNull(utility_matrix[test_user])):
                counter_book += 1
                print(str(score_similarity_book[j][1]+1) + " with similarity score = " + str(
                    score_similarity_book[j][0]))
            if counter_book == 5:
                break


print("load pickle file finished")
number_user = len(utility_matrix)
number_item = len(utility_matrix[0])
print("number user = " + str(number_user))
print("number item = " + str(number_item))
similarities = ['cosine']
for s in similarities: user_based(s)
