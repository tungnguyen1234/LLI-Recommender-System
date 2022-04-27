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
from tensor_retrieve import *

def extract_3D_dataset(limit = None):
    '''
    Desciption:
        Extracts the age, occupation, and gender features from the users
    Input:
        limit: int 
            The limit amount of data that would be process. Default is None, meaning having no limit
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
    genders = df['Gender'].to_numpy()

    return ages, occupations, genders


def tensor_traintest_score(tensor, percent, epsilon): 
    '''
    Desciption:
        This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
        recommendation result as the maximum value by the feature dimension for each (user, product) pair 
        as max(tensor[user, product, :]) 
    Input:
        tensor: np.array 
            The tensor of user ratings on films based on different features
        percent: int
            The percentage of splitting for training and testing data. This value ranges from 
            0 to 1 and default is None.
    Output:
        Returns the MAE, MSE and errors from the latent scaling convergence steps.
    '''

    if not (0<= percent <1):
        percent = 1

    users, films, features = np.nonzero(tensor)
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
        film = films[test[i]]
        feature = features[test[i]]
        rating = tensor[user, film, feature]
        re_train[(user, film)] = rating
        tensor[user, film, feature] = 0


    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(tensor, epsilon)

    # Test
    MAE = 0
    MSE = 0
    for i in range(len(test)):
        user = users[test[i]]
        film = films[test[i]]
        feature = features[test[i]]
        rating = 1/(latent_user[user]*latent_prod[film]*latent_feature[feature])
        re_test[(user, film)] = max(re_train[(user, film)], rating)
        
    for key, rating in re_test.items():
        diff = abs(re_train[key] - re_test[key])
        MAE += diff
        MSE += diff**2

    re_train = np.array(re_train)
    re_test = np.array(re_test)
    MAE = float(MAE/len(test))
    MSE = float(MSE/len(test))
    return MAE, MSE, errors


def tensor_movieLens(feature_vector, percent, limit, epsilon):
    '''
    Desciption:
        This function runs all the steps to pre-processing MovieLens data, running the tensor latent
        algorithm, and retrieving the MAE and MSE score. 
    Input:
        percent: int
            The percentage of splitting for training and testing data. Default is None.
        limit: int 
            The limit amount of data that would be process. Default is None, meaning having no limit
        limit: int 
            The limit amount of data that would be process. Default is None, meaning having no limit
        feature_vector: List[str]
            The features by string that would be added in the third dimension. There are three types:
            age, occupation, and gender.
    Output:
        Prints the MAE and MSE score.
    '''


    ages, occupations, genders = extract_3D_dataset(limit)
    matrix_rating = matrix_construct()

    # Testing purpose
    matrix_rating = np.array([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
    ages = np.array([1, 20, 30])
    occupations = np.array([0, 4, 5, 6])

    tensor_rating = np.array([])
    if len(feature_vector) == 1:
        feature = feature_vector[-1]
        if feature == 'age':
            tensor_rating = tensor_age(matrix_rating, ages)
        if feature == 'occup':
            tensor_rating = tensor_occupation(matrix_rating, occupations)
        if feature == 'gender':
            tensor_rating = tensor_gender(matrix_rating, genders)
    elif len(feature_vector) == 2:
        if any(_ in feature_vector for _ in ['age', 'occup']):
            tensor_rating = tensor_age_occup(matrix_rating, ages, occupations)
    
    
    MAE, MSE, errors = tensor_traintest_score(tensor_rating, percent, epsilon)
    print("MAE is", round(MAE, 2))
    print("MSE is", round(MSE, 2))
    print("Errors from the iteration process is:\n", errors)
