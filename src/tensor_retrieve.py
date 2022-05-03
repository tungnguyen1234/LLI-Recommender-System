#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import pandas as pd
import numpy as np
import torch as t
from math import *


def tensor_construct(device, matrix_rating, features, ages, occupations, genders):
    '''
    Desciption:
        Extracts the tensor from matrix_rating depending on the feature vectors for the third dimension
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        features: set
            The feature categories
    Output:
        Returns the tensor having the rating by (user, product, feature) tuple
    '''

    tensor_rating = t.tensor([])
    if len(features) == 1:
        if 'age' in features:
            tensor_rating = tensor_age(device, matrix_rating, ages)
        if 'occup' in features:
            tensor_rating = tensor_occupation(device, matrix_rating, occupations)
        if 'gender' in features:
            tensor_rating = tensor_gender(device, matrix_rating, genders)
    elif len(features) == 2:
        if features == set(['age', 'occup']):
            tensor_rating = tensor_age_occup(device, matrix_rating, ages, occupations)
        if features == set(['age', 'gender']):
            tensor_rating = tensor_age_gender(device, matrix_rating, ages, genders)
        if features == set(['gender', 'occup']):
            tensor_rating = tensor_gender_occup(device, matrix_rating, genders, occupations)
    else:
        tensor_rating = tensor_all(device, matrix_rating, ages, genders, occupations)

    return tensor_rating


def tensor_age(device, matrix_rating, ages):
    '''
    Desciption:
        Extracts the tensor having ages as the third dimension. We construct the tensor
        by projecting the matrix rating of (user, product) pair into the respective tuple
        (user, product, age) in the tensor
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        ages: t.tensor
            The ages of the users
    Output:
        Returns the tensor having the rating by (user, product, age) tuple
    '''

    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape

    # For Age: from 0 to 56 -> group 1 to 6. 
    third_dim = int(max(ages)/10) + 1
    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < len(ages):
            age = int(ages[user]/10)      
            tensor_rating[user, product, age] = matrix_rating[user, product]
    return tensor_rating 


def tensor_occupation(device, matrix_rating, occupations):
    '''
    Desciption:
        Extracts the tensor having occupations as the third dimension. We construct the tensor
        by projecting the matrix rating of (user, product) pair into the respective tuple
        (user, product, occupation) in the tensor
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        occupcations: t.tensor
            The occupations of the users
        
    Output:
        Returns the tensor having the rating by (user, product, occupation) tuple
    '''


    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    third_dim = max(occupations) + 1
    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < len(occupations):
            occup = occupations[user]         
            tensor_rating[user, product, occup] = matrix_rating[user, product]
    return tensor_rating 



def tensor_gender(device, matrix_rating, genders):
    '''
    Desciption:
        Extracts the tensor having genders as the third dimension. We construct the tensor
        by projecting the matrix rating of (user, product) pair into the respective tuple
        (user, product, gender) in the tensor
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        genders: t.tensor
            The genders of the users
        
    Output:
        Returns the tensor having the rating by (user, product, gender) tuple
    '''


    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    third_dim = max(genders) + 1
    

    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < len(genders):
            gender = genders[user]         
            tensor_rating[user, product, gender] = matrix_rating[user, product]
    return tensor_rating 

def tensor_age_occup(device, matrix_rating, ages, occupations):
    '''
    Desciption:
        Extracts the tensor having ages and occupation as the third dimension. We construct the tensor
        by projecting the matrix rating of (user, product) pair into the respective tuples
        (user, product, age) and (user, product, occupation) in the tensor.
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        ages: t.tensor
            The ages of the users
        occupcations: t.tensor
            The occupations of the users
        
    Output:
        Returns the tensor having the rating by (user, product, age) 
                                and (user, product, occupation) tuples
    '''

    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    # First group by occupation then age.
    third_dim = max(occupations) + 1 + int(max(ages)/10) + 1

    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < min(len(occupations), len(ages)):
            occup = occupations[user]     
            age = max(occupations) + 1 + int(ages[user]/10)

            tensor_rating[user, product, occup] = matrix_rating[user, product]
            tensor_rating[user, product, age] = matrix_rating[user, product]
    return tensor_rating 


def tensor_age_gender(device, matrix_rating, genders, ages):
    '''
    Desciption:
        Extracts the tensor having genders and ages as the third dimension. We construct the tensor
        by projecting the matrix rating of (user, product) pair into the respective tuples
        (user, product, gender) and (user, product, age) in the tensor.
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        genders: t.tensor
            The genders of the users
        ages: t.tensor
            The ages of the users
        
    Output:
        Returns the tensor having the rating by (user, product, gender) and (user, product, age) tuples
    '''


    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    third_dim = int(max(ages)/10) + 1 + max(genders) + 1
    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < min(len(genders), len(ages)):
            age = int(ages[user]/10)  
            gender = int(max(ages)/10) + 1 + genders[user]      
            tensor_rating[user, product, age] = matrix_rating[user, product]
            tensor_rating[user, product, gender] = matrix_rating[user, product]
    return tensor_rating 


def tensor_gender_occup(device, matrix_rating, genders, occupations):
    '''
    Desciption:
        Extracts the tensor having genders and occupations as the third dimension. 
        We construct the tensor by projecting the matrix rating of (user, product) pair into 
        the respective tuples (user, product, gender) and (user, product, occupation) in the tensor
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        genders: t.tensor
            The genders of the users
        occupations: t.tensor
            The occupations of the users
        
    Output:
        Returns the tensor having the rating by (user, product, gender) and (user, product, occupation) tuples
    '''


    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    third_dim = max(genders) + 1 + max(occupations) + 1 
    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < min(len(genders), len(occupations)):
            gender = genders[user]
            occup = max(genders) + 1 + occupations[user]         
            tensor_rating[user, product, occup] = matrix_rating[user, product]
            tensor_rating[user, product, gender] = matrix_rating[user, product]
    return tensor_rating 



def tensor_all(device, matrix_rating, ages, genders, occupations):
    '''
    Desciption:
        Extracts the tensor having ages, genders, and occupations as the third dimension. 
        We construct the tensor by projecting the matrix rating of (user, product) pair into 
        the respective tuples (user, product, gender), (user, product, age), 
        and (user, product, occupation) in the tensor
    Input:
        matrix_rating: t.tensor 
            The matrix of user prediction on products
        genders: t.tensor
            The genders of the users
        occupations: t.tensor
            The occupations of the users
        ages: t.tensor
            The ages of the users
        
    Output:
        Returns the tensor having the rating by (user, product, gender), (user, product, age)
        and (user, product, occupation) tuples
    '''


    # Get the nonzero for faster process
    user_product = t.nonzero(matrix_rating).to(device)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    third_dim = int(max(ages)/10) + 1 + max(genders) + 1 + max(occupations) + 1 
    tensor_rating = t.zeros((first_dim, second_dim, third_dim)).to(device)
  
    for user, product in user_product:
        if user < min(len(genders), len(occupations), len(ages)):
            age = int(ages[user]/10) 
            gender = int(max(ages)/10) + 1 + genders[user]     
            occup =  int(max(ages)/10) + 1 + max(genders) + 1 + occupations[user]
            tensor_rating[user, product, age] = matrix_rating[user, product]
            tensor_rating[user, product, gender] = matrix_rating[user, product]
            tensor_rating[user, product, occup] = matrix_rating[user, product]
    return tensor_rating 


