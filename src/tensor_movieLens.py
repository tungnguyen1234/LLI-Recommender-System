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
from tensor_3D_latent import tensor_latent



def tensor_movieLens(device, features, percent, limit, epsilon):
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
    MAEs = []
    RMSEs = []
    list_errors = []
    
    ages, occupations, genders = extract_features(device, limit)
    matrix_rating = matrix_construct(device)

    print("The algorithm runs 2 times to get the mean and std!")
    for i in range(2):
        print("-------------------------------------------------")
        print(f"Step {i+1}:")
        MAE, RMSE, errors = tensor_traintest_score(device, tensor_rating, percent, epsilon)
        MAE = float(MAE)
        RMSE = float(RMSE)
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        list_errors.append(errors)
    
    print("-------------------------------------------------")    
    print("MAE list is", MAEs)
    print("RMSE list is", RMSEs)
    print("Errors from the iteration process is:\n", list_errors)

    

    # Testing purpose
    # matrix_rating = t.tensor([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
    # ages = t.tensor([1, 20, 30])
    # occupations = t.tensor([0, 4, 5])
    # genders = t.tensor([0, 1, 0])
    
    tensor_rating = tensor_construct(device, matrix_rating, features, ages, occupations, genders)
    
    print("MAE is", round(MAE, 2))
    print("RMSE is", round(RMSE, 2))
    print("Errors from the iteration process is:\n", errors)


def extract_features(device, limit = None):
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
    ages = t.tensor(df['Age'].to_numpy()).to(device)

    # Job
    occupations = t.tensor(df['Occupation'].to_numpy()).to(device)

    # Gender
    genders = []
    for gender in list(df['Gender']):
        if gender == 'F':
            genders.append(0)
        elif gender == 'M':
            genders.append(1)

    genders = t.tensor(genders).to(device)
    return ages, occupations, genders


def tensor_traintest_score(device, tensor, percent, epsilon): 
    '''
    Desciption:
        This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
        recommendation result as the maximum value by the feature dimension for each (user, product) pair 
        as max(tensor[user, product, :]) 
    Input:
        tensor: torch.tensor 
            The tensor of user ratings on products based on different features
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

    user_prod_feat = t.nonzero(tensor).to(device)
    per = t.randperm(len(user_prod_feat)).to(device)
    # Get random test by percent
    num_test = int(percent*len(user_prod_feat))
    test = per[:num_test]

    re_train = {}
    re_test = {}

    # Setup
    for i in range(len(test)):
        user, product, feature = user_prod_feat[test[i]]
        rating = tensor[user, product, feature].clone()
        re_train[(int(user), int(product))] = rating
        tensor[user, product, feature] = 0

    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(device, tensor, epsilon)

    # Test
    MAE = 0.0
    MSE = 0.0
    for i in range(len(test)):
        user, product, feature = user_prod_feat[test[i]]
        rating = 1/(latent_user[user]*latent_prod[product]*latent_feature[feature])
        comp = re_train[(int(user), int(product))]
        re_test[int(user), int(product)] = t.max(comp, rating).to(device)
        
    for key, rating in re_test.items():
        diff = float(abs(re_train[key] - re_test[key]))
        MAE += diff
        MSE += diff**2

    MAE = MAE/len(test)
    RMSE = np.sqrt(MSE/len(test))
    return MAE, RMSE, errors
