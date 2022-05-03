#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import pandas as pd
from math import *
import torch as t


def matrix_latent(device, matrix, epsilon = 1e-10):  
    '''
    Desciption:
        This function runs the matrix latent invariant algorithm.
    Input:
        tensor: torch.tensor
            The tensor to retrive the latent variables from. 
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the latent vectors and the convergent errors from the iterative steps.
    '''
    matrix = matrix.to(device)
    m, n = matrix.shape
    # get the number of zeros in each dimension
    sigma_row = t.zeros(m).to(device)
    sigma_col = t.zeros(n).to(device)
    # Create a mask of non-zero elements
    rho_sign = (matrix != 0)*1
        
    # Get the number of nonzeros inside each row
    sigma_row = t.sum(rho_sign, 1)
    sigma_col = t.sum(rho_sign, 0)

    # Take log spaceof matrix
    matrix_log = t.log(matrix)
    
    # After log, all 0 values will be -inf, so we set them to 0
    matrix_log[matrix_log == - float('inf')] = 0.0
  
    # Initiate lantent variables
    latent_u = t.zeros((m, 1)).to(device)
    latent_p = t.zeros((n, 1)).to(device)
    
    # get the errors after each iteration
    errors = []

    # Starting the iterative steps
    trial = 0

    while True:
        error = 0.0
        for row in range(0,m):
            # Get the sum by rows first
            sig_size = sigma_row[row]
            # Update rho_row
            if sig_size > 0:
                rho_row = - t.sum(matrix_log[row, :])/sig_size
                matrix_log[row, :] += rho_row*rho_sign[row, :]
                latent_u[row] += rho_row
                error += float(rho_row**2)
               
        for col in range(0,n):
            # Get the sum by columns first
            sig_size = sigma_col[col]
            # Update rho_col
            if sig_size > 0:
                rho_col = - t.sum(matrix_log[:, col])/sig_size
                matrix_log[:, col] += rho_col*rho_sign[:, col]
                latent_p[col] += rho_col
                error += float(rho_col**2)

        trial += 1
        print('This is my', trial, 'time with error', error)
        errors.append(round(error, 20))
        if error <= epsilon:
            break

    # return the latent variables and errors
    return t.exp(latent_u), t.exp(latent_p), errors


