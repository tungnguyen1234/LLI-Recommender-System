#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace, 
import pandas as pd
import numpy as np
from numpy import *
from math import *
from matrix_movieLens import matrix_rating
from tensor_3D_latent import tensor_latent

def extract_3D_dataset(limit = None):
    '''
    Desciption:
        Extracts the age and occupation features
    Input:
        limit: TODO: give the type of limit (int, bool, string?) and describe what it does
    Output:
        similar to input
    '''
    
    # csv_movies = pd.read_csv('movies.csv')
    csv_users = pd.read_csv('users.csv', names = ["UserID", "Gender","Age","Occupation", "Zip-code"])
    df = pd.DataFrame(csv_users)
    df = df.head(limit) if limit else df.head()
    # Get age and profile info
    ages = df['Age'].to_numpy()
    # Job
    occupations = df['Occupation'].to_numpy()

    # Gender
    # gender = df['Gender'].to_numpy()

    return ages, occupations



''' This function constructs the tensor with third dimension as occupation and 6 ranges of ages.'''
def tensor_age_occup(matrix_rating, ages, occupations):
    # Get the nonzero for faster process
    idxusers, idxfilms = np.nonzero(matrix_rating)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    # The third one is tricky: First occupation then age.
    # For Age: from 0 to 56 -> group 1 to 6. 
    # For Occupation:  20 0
    third_dim = max(occupations) + 1 + int(max(ages)/10) + 1
    
    # Only occupation or age
    # third_dim = max(occupations)
    # third_dim = max(age)//10

    tensor_rating = zeros((first_dim, second_dim, third_dim))
  
    for i in range(len(idxusers)):
        # set at age
        user = idxusers[i]
        film = idxfilms[i]
        # Occupation as job or gender, as long as the index starts with 0
        if user < len(occupations) and user < len(ages):
            occup = occupations[user]     
            age = occup + 1 + int(ages[user]/10)

            tensor_rating[user, film, occup] = matrix_rating[user, film]
            tensor_rating[user, film, age] = matrix_rating[user, film]
    return tensor_rating 


''' This function onstructs the tensor with third dimension as 6 ranges of ages'''

def tensor_age(matrix_rating, ages):
    # Get the nonzero for faster process
    idxusers, idxfilms = np.nonzero(matrix_rating)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    # Only age
    third_dim = int(max(ages)/10) + 1
    tensor_rating = zeros((first_dim, second_dim, third_dim))
  
    for i in range(len(idxusers)):
        user = idxusers[i]
        film = idxfilms[i]
        if user < len(occupations):
            age = int(ages[user]/10)      
            tensor_rating[user, film, age] = matrix_rating[user, film]
    return tensor_rating 


'''This function constructs the tensor with third dimension as occupation'''
def tensor_occupation(matrix_rating, occupations):
    # Get the nonzero for faster process
    idxusers, idxfilms = np.nonzero(matrix_rating)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    
    # Only occupation
    third_dim = max(occupations) + 1
    tensor_rating = zeros((first_dim, second_dim, third_dim))
  
    for i in range(len(idxusers)):
        # set at age
        user = idxusers[i]
        film = idxfilms[i]
        # Occupation as job or gender, as long as the index starts with 0
        if user < len(occupations):
            occup = occupations[user]         
            tensor_rating[user, film, occup] = matrix_rating[user, film]
    return tensor_rating 


'''
This function takes a tensor and percentage of train-test split to split a train tensor for 
latent scaling algorithm and a testing vector for comparision. It also calculates the MAE and MSE.
'''
def tensor_traintest_score(tensor, percent): 
    users, films, features = np.nonzero(tensor)
    per = np.random.permutation(range(len(users)))
    # Get random test by percent
    num_test = round(percent*len(users))
    test = per[:num_test]

    re_train = []
    re_test = []

    # Setup
    for i in range(len(test)):
        user = users[test[i]]
        film = films[test[i]]
        feature = features[test[i]]
        rating = tensor[user, film, feature]
        re_train.append(rating)
        tensor[user, film, feature] = 0


    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(tensor)

    # Test
    MAE = 0
    MSE = 0
    for i in range(len(test)):
        user = users[test[i]]
        film = films[test[i]]
        feature = features[test[i]]
        rating = 1/(latent_user[user]*latent_prod[film]*latent_feature[feature])
        diff = abs(re_train[i] - rating)
        re_test.append(rating)
        MAE += diff
        MSE += diff**2

    re_train = np.array(re_train)
    re_test = np.array(re_test)
    MAE = float(MAE/len(test))
    MSE = float(MSE/len(test))
    return MAE, MSE, errors
