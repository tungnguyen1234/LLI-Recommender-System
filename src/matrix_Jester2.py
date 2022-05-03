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



def matrix_Jester2(device, percent, epsilon):
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
    dataname = 'jester' 
    other_method = 'LLI'
    path = "result/"
    output_text = path + str(dataname) + ".txt"

    matrix_rating = matrix_construct(device)
    MAEs = []
    RMSEs = []
    list_errors = []
    
    print("The algorithm runs 2 times to get the mean and std!")
    for i in range(2):
        print("-------------------------------------------------")
        print(f"Step {i+1}:")
        MAE, RMSE, errors = matrix_traintest_score(device, matrix_rating, percent, epsilon)
        MAE = float(MAE)
        RMSE = float(RMSE)
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        list_errors.append(errors)
    
    
    meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
    meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
    print(f"MAE has mean {meanMAE} and std {stdMAE}")
    print(f"RMSE has mean {meanRMSE} and std {stdRMSE}")

    lines = [f"Here is the result of dataset {dataname} for {other_method} method",\
            "---------------------------------", \
            f"MAE has mean {meanMAE} and std {stdMAE}",\
            f"RMSE has mean {meanRMSE} and std {stdRMSE}", \
            "---------------------------------", "\n"]
    with open(output_text, "a", encoding='utf-8') as f:
        f.write('\n'.join(lines))

def matrix_construct(device):
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
    matrix_rating = t.tensor(matrix_rating, dtype = t.float).to(device)

    # Change 99 into 0 and rescale the matrix by the fill value 
    observed_matrix = (matrix_rating != 99)*1
    fill_value = t.abs(t.min(matrix_rating)) + 1
    matrix_rating = matrix_rating + t.full(matrix_rating.shape, fill_value = fill_value)
    matrix_rating = t.mul(matrix_rating, observed_matrix).to(device)


    return matrix_rating
