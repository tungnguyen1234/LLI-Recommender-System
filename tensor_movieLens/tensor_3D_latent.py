import pandas as pd
import numpy as np
from numpy import *
from math import *


'''
This function takes a tensor and perform tensor latent scaling algorithm. It returns the latent
vectors and the convergent errors after the iteration.
'''

def tensor_latent(tensor, epsilon = 1e-15):
    d1, d2, d3 = tensor.shape

    # Get the total number of nonzeros
    rho_sign = (tensor != 0)*1
    # Get number of nonzeros in each dimension
    sigma_first = zeros(d1)  
    sigma_second = zeros(d2)  
    sigma_third = zeros(d3)

    # Get the number of nonzeros inside each 2-dimensional tensor, meaning two “:s”
    for first in range(d1):
        sigma_first[first] = sum(rho_sign[first, :, :])

    for second in range(d2):
        sigma_second[second] = sum(rho_sign[:, second, :]) 

    for third in range(d3):
        sigma_third[third] = sum(rho_sign[:, :, third])

    # Take logarithm of tensor
    tensor_log = np.log(tensor)
    # After log, all 0 values will be -inf, so we set them to 0
    tensor_log[tensor_log == -Inf] = 0.0

    # Initiate convergence 
    latent_1 = zeros(d1)
    latent_2 = zeros(d2)
    latent_3 = zeros(d3)

    # Iteration errors
    errors = []

    trial = 0
    print('Start the scaling process')

    while True:
        error = 0

        for second in range(d2):
            # Get the sum by second dim
            sig_size = sigma_second[second]
            if sig_size > 0:
                # Update rho_second
                rho_second = - sum(tensor_log[:, second, :])/sig_size
                tensor_log[:, second, :] += rho_second*rho_sign[:, second, :]
                latent_2[second] += rho_second
                error += rho_second**2



        # Starting the first iterative step 
        for first in range(d1):
            # Get the sum by first dim
            sig_size = sigma_first[first]
            if sig_size > 0:
                # Update rho_first
                rho_first = - sum(tensor_log[first, :, : ])/sig_size
                tensor_log[first, :, :] += rho_first*rho_sign[first, :, :]
                latent_1[first] += rho_first
                error += rho_first**2
            
        for third in range(d3):
            # Get the sum by third dim
            sig_size = sigma_third[third]
            if sig_size > 0:
                # Update rho_third
                rho_third = - sum(tensor_log[:, :, third])/sig_size
                tensor_log[:, :, third] += rho_third*rho_sign[:, :, third]
                latent_3[third] += rho_third
                error += rho_third**2
    

        errors.append(error)
        trial += 1
        print('This is my ', trial, ' time with error', error)
        if error < epsilon:
            break


    return np.exp(latent_1), np.exp(latent_2), np.exp(latent_3), errors