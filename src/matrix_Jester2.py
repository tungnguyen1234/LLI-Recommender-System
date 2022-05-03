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



def matrix_Jester2(percent, epsilon):
    '''
    Desciption:
        This function runs all the steps to pre-processing jester2 data, running the tensor latent
        algorithm, and retrieving the MAE and RMSE score. 
    Input:
        percent: int
            The percentage of splitting for training and testing data. Default is None.
        limit: int 
            The limit amount of data that would be process. Default is None, meaning having no limit
    Output:
        Prints the MAE and RMSE score.
    '''

    matrix_rating = matrix_construct()
    MAE, RMSE, errors = matrix_traintest_score(matrix_rating, percent, epsilon)
    print("MAE of LLI Jester 2 is", t.round(MAE, decimals = 2))
    print("RMSE of LLI Jester 2 is", t.round(RMSE, decimals = 2))
    print("Errors from the iteration process is:\n", np.array(errors))  



def matrix_construct():
    '''
    Desciption:
        Gather csv files of Jester2 to retrievie the the numpy matrix of user-rating
    Output:
       A numpy matrix of user-rating
    '''

    ratings = pd.read_csv('data/jester.csv')
    # Eliminate the first column
    ratings = ratings.iloc[:,1:]
    matrix_rating = pd.DataFrame(ratings).to_numpy()    
    matrix_rating = t.tensor(matrix_rating, dtype = t.float)

    # Change 99 into 0 and rescale the matrix by the fill value 
    observed_matrix = (matrix_rating != 99)*1
    fill_value = t.abs(t.min(matrix_rating)) + 1
    matrix_rating = matrix_rating + t.full(matrix_rating.shape, fill_value = fill_value)
    matrix_rating = t.mul(matrix_rating, observed_matrix)


    return matrix_rating
