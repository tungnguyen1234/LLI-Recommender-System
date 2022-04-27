import pandas as pd
import numpy as np
from numpy import *
from math import *
from matrix_latent import matrix_latent


'''
This function gathers csv files of MovieLens1M and return the numpy matrix of user-rating
'''
def matrix_construct():
    ratings = pd.read_csv('ratings.csv', names = ["UserID", "MovieID","Rating","Timestamp"])
    df = pd.DataFrame(ratings)    
    sort_rating = df.sort_values(by = ['UserID', 'MovieID'], ignore_index = True)
    sort_rating_fill_0 = sort_rating.fillna(0)
    matrix_rating = sort_rating_fill_0.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
    matrix_rating = matrix_rating.to_numpy()
    
    return matrix_rating

'''
This function takes a matrix and percentage of train-test split to split a train matrix for 
latent scaling algorithm and a testing vector for comparision. It also calculates the MAE and MSE.
'''
def matrix_traintest_score(matrix, percent = None):
    if not percent:
        percent = 1

    users, films = np.nonzero(matrix)
    per = np.random.permutation(range(len(users)))
    num_test = int(percent*len(users))
    test = per[:num_test]

    re_train = []
    re_test = []

    # Setup
    for i in range(len(test)):
        user = users[test[i]]
        film = films[test[i]]
        rating = matrix[user, film]
        re_train.append(rating)
        matrix[user, film] = 0
    # Scaling
    latent_user, latent_prod, errors = matrix_latent(matrix)
    # Test
    MAE = 0
    MSE = 0
    for i in range(len(test)):
        user = users[test[i]]
        film = films[test[i]]
        rating = 1/(latent_user[user]*latent_prod[film])
        diff = abs(re_train[i] - rating)
        re_test.append(rating)
        MAE += diff
        MSE += diff**2

    MAE = float(MAE/len(test))
    MSE = float(MSE/len(test))
    return MAE, MSE, errors


def matrix_movieLens(percent):
    matrix_rating = matrix_construct()
    MAE, MSE, errors = matrix_traintest_score(matrix_rating, percent)
    print("MAE is", round(MAE, 2))
    print("MSE is", round(MSE, 2))
    print("Errors from the iteration process is:\n", errors)