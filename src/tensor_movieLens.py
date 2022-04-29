#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace
import pandas as pd
import numpy as np
from numpy import *
from math import *
from matrix_movieLens import matrix_construct
from tensor_3D_latent import tensor_latent
from tensor_retrieve import tensor_construct



def tensor_movieLens(features, percent, limit, epsilon):
    '''
    Desciption:
        This function runs all the steps to pre-processing MovieLens data, running the tensor latent
        algorithm, and retrieving the MAE and MSE score. 
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
        Prints the MAE and MSE score.
    '''


    ages, occupations, genders = extract_features(limit)
    matrix_rating = matrix_construct()

    # Testing purpose
    # matrix_rating = np.array([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
    # ages = np.array([1, 20, 30])
    # occupations = np.array([0, 4, 5])
    # genders = np.array([0, 1, 0])
    
    tensor_rating = tensor_construct(matrix_rating, features, ages, occupations, genders)
    MAE, MSE, errors = tensor_traintest_score(tensor_rating, percent, epsilon)
    print("MAE is", round(MAE, 2))
    print("MSE is", round(MSE, 2))
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
    
    csv_users = pd.read_csv('users.csv', names = ["UserID", "Gender","Age","Occupation", "Zip-code"])
    df = pd.DataFrame(csv_users)
    if limit:
        df = df.head(limit)

    # Get age and profile info
    ages = df['Age'].to_numpy()

    # Job
    occupations = df['Occupation'].to_numpy()

    # Gender
    genders = []
    for gender in list(df['Gender']):
        if gender == 'F':
            genders.append(0)
        elif gender == 'M':
            genders.append(1)

    genders = np.array(genders)
    return ages, occupations, genders


def tensor_traintest_score(tensor, percent, epsilon): 
    '''
    Desciption:
        This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
        recommendation result as the maximum value by the feature dimension for each (user, product) pair 
        as max(tensor[user, product, :]) 
    Input:
        tensor: np.array 
            The tensor of user ratings on products based on different features
        percent: int
            The percentage of splitting for training and testing data. This value ranges from 
            0 to 1 and default is None.
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the MAE, MSE and errors from the latent scaling convergence steps.
    '''

    if not (0<= percent <1):
        percent = 1

    users, products, features = np.nonzero(tensor)
    per = np.random.permutation(range(len(users)))
    # Get random test by percent
    percent = min(1, percent)
    num_test = int(percent*len(users))
    test = per[:num_test]

    re_train = {}
    re_test = {}

    # Setup
    for i in range(len(test)):
        user = users[test[i]]
        product = products[test[i]]
        feature = features[test[i]]
        rating = tensor[user, product, feature]
        re_train[(user, product)] = rating
        tensor[user, product, feature] = 0


    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(tensor, epsilon)

    # Test
    MAE = 0
    MSE = 0
    for i in range(len(test)):
        user = users[test[i]]
        product = products[test[i]]
        feature = features[test[i]]
        rating = 1/(latent_user[user]*latent_prod[product]*latent_feature[feature])
        re_test[(user, product)] = max(re_train[(user, product)], rating)
        
    for key, rating in re_test.items():
        diff = abs(re_train[key] - re_test[key])
        MAE += diff
        MSE += diff**2

    re_train = np.array(re_train)
    re_test = np.array(re_test)
    MAE = float(MAE/len(test))
    MSE = float(MSE/len(test))
    return MAE, MSE, errors


