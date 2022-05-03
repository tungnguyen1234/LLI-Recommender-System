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



def matrix_movieLens(device, percent, epsilon):
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
    dataname = 'ml-1m' 
    other_method = 'LLI'
    path = "result/"
    output_text = path + str(dataname) + ".txt"

    matrix_rating = matrix_construct(device)

    # Testing purpose:
    # matrix_rating = t.tensor([[0,1,2,3,4,1,1,0,0,0,0], \
    #  [0,2,0,1,0,0, 4, 7, 8, 9, 10]], dtype=t.float)
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
        Gather csv files of MovieLens to retrievie the the numpy matrix of user-rating
    Output:
       A numpy matrix of user-rating
    '''

    ratings = pd.read_csv('data/ratings.csv', names = ["UserID", "MovieID","Rating","Timestamp"])
    df = pd.DataFrame(ratings)    
    sort_rating = df.sort_values(by = ['UserID', 'MovieID'], ignore_index = True)
    sort_rating_fill_0 = sort_rating.fillna(0)
    matrix_rating = sort_rating_fill_0.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
    matrix_rating = t.tensor(matrix_rating.to_numpy(), dtype = t.float).to(device)
    
    return matrix_rating