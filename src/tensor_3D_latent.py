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

def tensor_latent(device, tensor, epsilon = 1e-10):
    '''
    Desciption:
        This function runs the tensor latent invariant algorithm.
    Input:
        tensor: torch.tensor
            The tensor to retrive the latent variables from. 
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the latent vectors and the convergent errors from the iterative steps.
    '''
    tensor = tensor.to(device)
    d1, d2, d3 = tensor.shape

    # Get the total number of nonzeros
    rho_sign = (tensor != 0)*1
    # Get number of nonzeros in each dimension
    sigma_first = t.zeros(d1).to(device)  
    sigma_second = t.zeros(d2).to(device)  
    sigma_third = t.zeros(d3).to(device)

    # Get the number of nonzeros inside each 2-dimensional tensor
    for first in range(0,d1):
        sigma_first[first] = t.sum(rho_sign[first, :, :])

    for second in range(0,d2):
        sigma_second[second] = t.sum(rho_sign[:, second, :]) 

    for third in range(0,d3):
        sigma_third[third] = t.sum(rho_sign[:, :, third])

    # Take logarithm of tensor
    tensor_log = t.log(tensor)
    # After log, all 0 values will be -inf, so we set them to 0
    tensor_log[tensor_log == - float("Inf")] = 0.0

    # Initiate convergence 
    latent_1 = t.zeros(d1).to(device)
    latent_2 = t.zeros(d2).to(device)
    latent_3 = t.zeros(d3).to(device)

    # Iteration errors
    errors = []

    trial = 0
    print('Start the scaling process')

    while True:
        error = 0.0

        # Update in second dim
        subtract_second = t.div(tensor_log.sum([0, 2]), sigma_second).nan_to_num(0.0)
        rho_second = - subtract_second # d2
        tensor_log += rho_second[None, :, None] * rho_sign # d2 - d1*d2*d3
        latent_2 += rho_second # latent_2 = rho_second
        error += (rho_second**2).sum()

        # Update in first dim
        subtract_first = t.div(tensor_log.sum([1, 2]), sigma_first).nan_to_num(0.0)
        rho_first = - subtract_first # d1
        tensor_log += rho_first[:, None, None] * rho_sign  # d1 - d1*d2*d3
        latent_1 += rho_first 
        error += (rho_first**2).sum()

        # Update in third dim
        subtract_third = t.div(tensor_log.sum([0, 1]), sigma_third).nan_to_num(0.0)
        rho_third =  - subtract_third # d3
        tensor_log += rho_third[None, None, :] * rho_sign # d3 - d1*d2*d3
        latent_3 += rho_third
        error += (rho_third**2).sum()

        errors.append(error)
        trial += 1
        print('This is my ', trial, ' time with error', error)
        if error < epsilon:
            break

    return np.exp(latent_1), np.exp(latent_2), np.exp(latent_3), errors




        # # Starting the first iterative step 
        # for first in range(0,d1):
        #     # Get the sum by first dim
        #     sig_size = sigma_first[first]
        #     if sig_size > 0:
        #         # Update rho_first
        #         rho_first = - t.sum(tensor_log[first, :, : ])/sig_size
        #         tensor_log[first, :, :] += rho_first*rho_sign[first, :, :]
        #         latent_1[first] += rho_first
        #         error += float(rho_first**2)
            
        # for third in range(0,d3):
        #     # Get the sum by third dim
        #     sig_size = sigma_third[third]
        #     if sig_size > 0:
        #         # Update rho_third
        #         rho_third = - t.sum(tensor_log[:, :, third])/sig_size
        #         tensor_log[:, :, third] += rho_third*rho_sign[:, :, third]
        #         latent_3[third] += rho_third
        #         error += float(rho_third**2)
    