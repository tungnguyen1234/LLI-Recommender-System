#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import numpy as np
import torch as t

class TensorLLI():
    def __init__(self, device, tensor, epsilon = 1e-10):
        self.device = device
        self.tensor = tensor
        self.epsilon = epsilon 

    def LLI(self):
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
        
        d1, d2, d3 = self.tensor.shape

        # Get the total number of nonzeros
        rho_sign = (self.tensor != 0)*1

        # Get the number of nonzeros inside each 2-dimensional tensor
        sigma_first = rho_sign.sum([1,2])
        sigma_second = rho_sign.sum([0,2])
        sigma_third = rho_sign.sum([0,1])

        # Take logarithm of tensor
        tensor_log = t.log(self.tensor)
        # After log, all 0 values will be -inf, so we set them to 0
        tensor_log[tensor_log == - float("Inf")] = 0.0

        # Initiate convergence 
        latent_1 = t.zeros(d1).to(self.device)
        latent_2 = t.zeros(d2).to(self.device)
        latent_3 = t.zeros(d3).to(self.device)

        # Iteration errors
        errors = []

        trial = 0
        print('Start the scaling process')

        while True:
            error = 0.0

            # Update in second dim
            rho_second = - t.div(tensor_log.sum([0, 2]), sigma_second).nan_to_num(0.0) #d2
            tensor_log += rho_second[None, :, None] * rho_sign # d2 - d1*d2*d3
            latent_2 += rho_second 
            error += (rho_second**2).sum()

            # Update in first dim
            rho_first = - t.div(tensor_log.sum([1, 2]), sigma_first).nan_to_num(0.0) # d1
            tensor_log += rho_first[:, None, None] * rho_sign  # d1 - d1*d2*d3
            latent_1 += rho_first 
            error += (rho_first**2).sum()

            # Update in third dim
            rho_third = - t.div(tensor_log.sum([0, 1]), sigma_third).nan_to_num(0.0) # d3
            tensor_log += rho_third[None, None, :] * rho_sign # d3 - d1*d2*d3
            latent_3 += rho_third
            error += (rho_third**2).sum()

            errors.append(float(error))
            trial += 1
            print('This is my', trial, 'time with error', float(error))
            if error < self.epsilon:
                break

        return t.exp(latent_1), t.exp(latent_2), t.exp(latent_3), errors