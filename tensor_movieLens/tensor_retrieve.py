
import pandas as pd
import numpy as np
from numpy import *
from math import *





def tensor_age_occup(matrix_rating, ages, occupations):
    '''
    Desciption:
        Extracts the tensor having age and occupation as the third dimension. We construct the tensor
        by project the matrix rating of (user, product) pair into the respective tuple
        (user, product, age) and (user, product, occupation) in the tensor.
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

    # First group by occupation then age.
    # For Age: from 0 to 56 -> group 1 to 6. 
    # For Occupation:  20 0
    third_dim = max(occupations) + 1 + int(max(ages)/10) + 1

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
        Extracts the tensor having age as the third dimension. We construct the tensor
        by project the matrix rating of (user, product) pair into the respective tuple
        (user, product, age) in the tensor.
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
        Extracts the tensor having age and occupation as the third dimension. We construct the tensor
        by project the matrix rating of (user, product) pair into the respective tuple
        (user, product, occupation) in the tensor.
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



def tensor_gender(matrix_rating, genders):
    '''
    Desciption:
        Extracts the tensor having age and genders as the third dimension. We construct the tensor
        by project the matrix rating of (user, product) pair into the respective tuple
        (user, product, gender) in the tensor.
    Input:
        matrix_rating: np.array 
            The matrix of user prediction on films
        genders: np.array
            The genders of the users
        
    Output:
        Returns the tensor having the rating by (user, product, gender) category
    '''


    # Get the nonzero for faster process
    idxusers, idxfilms = np.nonzero(matrix_rating)

    # Get the dimensions 
    first_dim, second_dim = matrix_rating.shape
    
    # Only occupation
    third_dim = max(genders) + 1
    tensor_rating = zeros((first_dim, second_dim, third_dim))
  
    for i in range(len(idxusers)):
        # set at age
        user = idxusers[i]
        film = idxfilms[i]
        # Occupation as job or gender, as long as the index starts with 0
        if user < len(genders):
            gender = genders[user]         
            tensor_rating[user, film, gender] = matrix_rating[user, film]
    return tensor_rating 

