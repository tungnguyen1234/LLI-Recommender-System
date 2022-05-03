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
from matrix_score_eval import matrix_traintest_score



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

        # Testing purpose:
        # matrix_rating = t.tensor([[0,1,2,3,4,1,1,0,0,0,0], \
        #  [0,2,0,1,0,0, 4, 7, 8, 9, 10]], dtype=t.float)
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