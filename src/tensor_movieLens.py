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
import os
from math import *
from tensor_retrieve import tensor_train_test
from tensor_3D_latent import tensor_latent
from matrix_movieLens import matrix_construct
from tqdm import tqdm



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
    '''
    MAEs = []
    RMSEs = []
    list_errors = []
    
    path = "result/"
    output_text = path + "tensor_ml-1m_result" + ".txt"
        
    ages, occupations, genders = extract_features(device, limit)
    matrix_rating = matrix_construct(device)
    # os.remove(output_text)


    print("The algorithm runs 2 times to get the mean and std!")
    for i in range(2):
        print("-------------------------------------------------")
        print(f"Step {i+1}:")
        MAE, RMSE, errors = tensor_score(device, matrix_rating, ages, occupations, genders, features, percent, epsilon)
        MAE = float(MAE)
        RMSE = float(RMSE)
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        list_errors.append(errors)
    print("-------------------------------------------------")    
    mean_errors = [np.mean([list_errors[0][i], list_errors[1][i]]) for i in range(len(list_errors[0]))]
    std_errors = [np.std([list_errors[0][i], list_errors[1][i]]) for i in range(len(list_errors[0]))]
    meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
    meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
    print(f"MAE has mean {meanMAE} and std {stdMAE}")
    print(f"RMSE has mean {meanRMSE} and std {stdRMSE}")


    
    lines = [f"MAE has mean {meanMAE} and std {stdMAE}", f"RMSE has mean {meanRMSE} and std {stdRMSE}",\
            f"Error means from the iteration process is {mean_errors}", \
            f"Error stds from the iteration process is {std_errors}",]
    with open(output_text, "a", encoding='utf-8') as f:
        f.write('\n'.join(lines))

    

    # Testing purpose
    # matrix_rating = t.tensor([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
    # ages = t.tensor([1, 20, 30])
    # occupations = t.tensor([0, 4, 5])
    # genders = t.tensor([0, 1, 0])


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


def tensor_score(device, matrix_rating, ages, occupations, genders, features, percent, epsilon): 
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

    tensor_rating, train_bag, test_bag = tensor_train_test(device, matrix_rating, features, ages, occupations, genders, percent)
    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(device, tensor_rating, epsilon)

    re_test = {}
    # Get the maximum rating for each user and product 
    for i in range(len(test_bag)):
        user, product, feature = test_bag[i]
        rating = 1/(latent_user[user]*latent_prod[product]*latent_feature[feature])
        val = matrix_rating[int(user), int(product)]
        re_test[(user, product)] = t.max(val, rating)
        
    
    # Regroup the ratings to get RMSE and MSE
    score_train = [] 
    score_test = []
    for key, rating in re_test.items():
        user, product = key
        score_train.append(matrix_rating[int(user), int(product)])
        score_test.append(rating)

    # Get RMSE and MSE
    mae_loss = t.nn.L1Loss()
    mse_loss = t.nn.MSELoss()
    score_train, score_test = t.tensor(score_train), t.tensor(score_test)
    RMSE = t.sqrt(mse_loss(score_train, score_test))
    MAE = mae_loss(score_train, score_test)

    return MAE, RMSE, errors
