#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import pandas as pd
import numpy as np
from numpy import *
from math import *


def matrix_latent(matrix, epsilon = 1e-10):  
    '''
    Desciption:
        This function runs the matrix latent invariant algorithm.
    Input:
        tensor: np.array
            The tensor to retrive the latent variables from. 
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the latent vectors and the convergent errors from the iterative steps.
    '''

    m, n = matrix.shape
    # get the number of zeros in each dimension
    sigma_row = zeros(m)
    sigma_col = zeros(n)
    # Create a mask of non-zero elements
    rho_sign = (matrix != 0)*1
        
    # Get the number of nonzeros inside each row
    sigma_row = sum(rho_sign, axis = 1)
    sigma_col = sum(rho_sign, axis = 0)

    # Take log spaceof matrix
    matrix_log = np.log(matrix)
    
    # After log, all 0 values will be -inf, so we set them to 0
    matrix_log[matrix_log == - Inf] = 0.0
  
    # Initiate lantent variables
    latent_u = np.zeros((m, 1))
    latent_p = np.zeros((n, 1))
    
    # get the errors after each iteration
    errors = []

    # Starting the iterative steps
    trial = 0

    while True:
        error = 0
        for row in range(m):
            # Get the sum by rows first
            sig_size = sigma_row[row]
            # Update rho_row
            if sig_size > 0:
                rho_row = - sum(matrix_log[row])/sig_size
                matrix_log[row] += rho_row*rho_sign[row]
                latent_u[row] += rho_row
                error += rho_row**2
               
        for col in range(n):
            # Get the sum by columns first
            sig_size = sigma_col[col]
            # Update rho_col
            if sig_size > 0:
                rho_col = - sum(matrix_log[:, col])/sig_size
                matrix_log[:, col] += rho_col*rho_sign[:, col]
                latent_p[col] += rho_col
                error += rho_col**2

        trial += 1
        print('This is my', trial, 'time with error', error)
        errors.append(round(error, 20))
        if error <= epsilon:
            break

    # return the latent variables and errors
    return np.exp(latent_u), np.exp(latent_p), errors


