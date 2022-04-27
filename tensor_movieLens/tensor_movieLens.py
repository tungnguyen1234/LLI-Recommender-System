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

def extract_3D_dataset(limit = None):
    '''
    Desciption:
        Extracts the age, occupation, and gender features from the users
    Input:
        limit: int 
            The limit amount of data that would be process. Default is None, meaning no limit to data
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



def tensor_age_occup(matrix_rating, ages, occupations):
    '''
    Desciption:
        Extracts the tensor having age and occupation as the third dimension.
    Input:
        matrix_rating: np.array 
            The matrix of user prediction on films
        ages: np.array
            The ages of the users
        occupcations: np.array
            The occupations of the users
        
    Output:
        Returns the tensor having the rating by (user, product, age) 
                                and (user, product, occupation) category
    '''

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


def tensor_age(matrix_rating, ages):
    '''
    Desciption:
        Extracts the tensor having age as the third dimension.
    Input:
        matrix_rating: np.array 
            The matrix of user prediction on films
        ages: np.array
            The ages of the users
    Output:
        Returns the tensor having the rating by (user, product, age) category
    '''

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
        if user < len(ages):
            age = int(ages[user]/10)      
            tensor_rating[user, film, age] = matrix_rating[user, film]
    return tensor_rating 


def tensor_occupation(matrix_rating, occupations):
    '''
    Desciption:
        Extracts the tensor having age and occupation as the third dimension.
    Input:
        matrix_rating: np.array 
            The matrix of user prediction on films
        occupcations: np.array
            The occupations of the users
        
    Output:
        Returns the tensor having the rating by (user, product, occupation) category
    '''


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
This function takes a tensor and percentage of train-test split to It also calculates the MAE and MSE.
'''
def tensor_traintest_score(tensor, percent = None): 
    '''
    Desciption:
        split a training tensor for latent scaling algorithm and a testing vector of ratings for comparison. 
    Input:
        tensor: np.array 
            The tensor of user prediction on films based on different features
        percent: int
            The percentage of splitting for training and testing data. Default is None.
    Output:
        Returns the tensor having the rating by (user, product, feature) category
    '''

    if not percent:
        percent = 1

    users, films, features = np.nonzero(tensor)
    per = np.random.permutation(range(len(users)))
    # Get random test by percent
    percent = min(1, percent)
    num_test = int(percent*len(users))
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


def tensor_movieLens(percent, limit, feature_vector):
    ages, occupations, genders = extract_3D_dataset(limit)
    # print(ages, occupations, genders)
    # matrix_rating = matrix_construct()
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
    elif len(feature_vector) == 2:
        if any(_ in feature_vector for _ in ['age', 'occup']):
            tensor_rating = tensor_age_occup(matrix_rating, ages, occupations)
    print(tensor_rating)
    MAE, MSE, errors = tensor_traintest_score(tensor_rating, percent)
    print("MAE is", round(MAE, 2))
    print("MSE is", round(MSE, 2))
    print("Errors from the iteration process is:\n", errors)
