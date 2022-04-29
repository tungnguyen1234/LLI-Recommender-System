#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import pandas as pd
import torch as t
import numpy as np
from math import *
from matrix_latent import matrix_latent



def matrix_movieLens(percent, epsilon):
    '''
    Desciption:
        This function runs all the steps to pre-processing MovieLens data, running the tensor latent
        algorithm, and retrieving the MAE and RMSE score. 
    Input:
        percent: int
            The percentage of splitting for training and testing data. Default is None.
        limit: int 
            The limit amount of data that would be process. Default is None, meaning having no limit
        feature_vector: List[str]
            The features by string that would be added in the third dimension. There are three types:
            age, occupation, and gender.
    Output:
        Prints the MAE and RMSE score.
    '''

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        matrix_rating = matrix_construct()
        MAE, RMSE, errors = matrix_traintest_score(matrix_rating, percent, epsilon)
        print("MAE is", np.round(MAE, 2))
        print("RMSE is", np.round(RMSE, 2))
        print("Errors from the iteration process is:\n", np.array(errors))  



def matrix_construct():
    '''
    Desciption:
        Gather csv files of MovieLens to retrievie the the numpy matrix of user-rating
    Output:
       A numpy matrix of user-rating
    '''

    ratings = pd.read_csv('data/ratings.csv', names = ["UserID", "MovieID","Rating","Timestamp"])
    df = pd.DataFrame(ratings)    
    sort_rating = df.sort_values(by = ['UserID', 'MovieID'], ignore_index = True)
    sort_rating_fill_0 = sort_rating.fillna(0)
    matrix_rating = sort_rating_fill_0.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
    matrix_rating = t.tensor(matrix_rating.to_numpy(), dtype = t.float)
    
    return matrix_rating



def matrix_traintest_score(matrix, percent, epsilon):
    '''
    Desciption:
        This function splits a training matrix for latent scaling algorithm and a testing vector to compare 
        and retrivie 
    Input:
        matrix: torch.tensor 
            The matrix of user rating on products.
        percent: int
            The percentage of splitting for training and testing data. This value ranges from 
            0 to 1 and default is None.
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the MAE, RMSE and errors from the latent scaling convergence steps.
    '''

    if not (0<= percent <1):
        percent = 1

    user_product = t.nonzero(matrix)
    per = t.randperm(len(user_product))
    num_test = int(percent*len(user_product))
    test = per[:num_test]

    re_train = []
    re_test = []

    # Setup
    for i in range(len(test)):
        user, product = user_product[test[i]]
        rating = matrix[user, product].clone()
        re_train.append(rating)
        matrix[user, product] = 0

    # Scaling
    latent_user, latent_prod, errors = matrix_latent(matrix, epsilon)
    # Test
    MAE = 0.0
    MSE = 0.0
    for i in range(len(test)):
        user, product = user_product[test[i]]
        rating = 1/(latent_user[user]*latent_prod[product])
        diff = float(abs(re_train[i] - rating))
        re_test.append(rating)
        MAE += diff
        MSE += diff**2

    MAE = MAE/len(test)
    RMSE = np.sqrt(MSE/len(test))
    return MAE, RMSE, errors
