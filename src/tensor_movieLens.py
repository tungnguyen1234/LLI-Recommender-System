#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace
import pandas as pd
import torch as t
import numpy as np
from math import *
from matrix_movieLens import matrix_construct
from tensor_retrieve import tensor_construct
from tensor_score_eval import tensor_traintest_score



def tensor_movieLens(features, percent, limit, epsilon):
    '''
    Desciption:
        This function runs all the steps to pre-processing MovieLens data, running the tensor latent
        algorithm, and retrieving the MAE and RMSE score. 
    Input:
        percent: int
            The percentage of splitting for training and testing data. Default is None.
        limit: int 
            The limit number of data that would be process. Default is None, meaning having no limit
        features: set()
            The features by string that would be added in the third dimension. There are three types:
            age, occupation, and gender.
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Prints the MAE, RMSE and errors from the latent scaling convergence steps.
 score.
    '''

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        ages, occupations, genders = extract_features(limit)
        matrix_rating = matrix_construct()

        # Testing purpose
        # matrix_rating = t.tensor([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
        # ages = t.tensor([1, 20, 30])
        # occupations = t.tensor([0, 4, 5])
        # genders = t.tensor([0, 1, 0])
        
        tensor_rating = tensor_construct(matrix_rating, features, ages, occupations, genders)
        MAE, RMSE, errors = tensor_traintest_score(tensor_rating, percent, epsilon)
        print("MAE of LLI MovieLens is", t.round(MAE, decimals = 2))
        print("RMSE of LLI MovieLens is", t.round(RMSE, decimals = 2))
        print("Errors from the iteration process is:\n", errors)


def extract_features(limit = None):
    '''
    Desciption:
        Extracts the age, occupation, and gender features from the users. Here we label gender 'F' as
        0 and gender 'M' as 1. The index of occupations are from 0 to 20, and the index of ages is from 1 to 56.
    Input:
        limit: int 
            The limit number of data that would be process. Default is None, meaning having no limit
    Output:
        Array of ages, occupation, and genders of the users
    '''
    
    csv_users = pd.read_csv('data/users.csv', names = ["UserID", "Gender","Age","Occupation", "Zip-code"])
    df = pd.DataFrame(csv_users)
    if limit:
        df = df.head(limit)

    # Get age and profile info
    ages = t.tensor(df['Age'].to_numpy())

    # Job
    occupations = t.tensor(df['Occupation'].to_numpy())

    # Gender
    genders = []
    for gender in list(df['Gender']):
        if gender == 'F':
            genders.append(0)
        elif gender == 'M':
            genders.append(1)

    genders = t.tensor(genders)
    return ages, occupations, genders

