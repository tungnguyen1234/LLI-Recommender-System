#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import torch as t
import gc

class TensorLLI():
    def __init__(self, device, dim, tensor, epsilon):
        self.dim = dim
        self.device = device
        self.tensor = tensor
        self.epsilon = epsilon 

    def LLI(self):
        if self.dim == 2:
            return self.LLI_2D()
        elif self.dim == 3:
            return self.LLI_3D()

    def LLI_3D(self):
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

        step = 1
        print('Start the LLI process:')

        while True:
            error = 0.0

            # Update in third dim
            rho_third = - t.div(tensor_log.sum([0, 1]), sigma_third).nan_to_num(0.0) # d3
            tensor_log += rho_third[None, None, :] * rho_sign # d3 - d1*d2*d3
            latent_3 -= rho_third
            error += (rho_third**2).sum()
            del rho_third
            
            # Update in first dim
            rho_first = - t.div(tensor_log.sum([1, 2]), sigma_first).nan_to_num(0.0) # d1
            tensor_log += rho_first[:, None, None] * rho_sign  # d1 - d1*d2*d3
            latent_1 -= rho_first 
            error += (rho_first**2).sum()
            del rho_first
            
            # Update in second dim
            rho_second = - t.div(tensor_log.sum([0, 2]), sigma_second).nan_to_num(0.0) #d2../
            tensor_log += rho_second[None, :, None] * rho_sign # d2 - d1*d2*d3
            latent_2 -= rho_second 
            error += (rho_second**2).sum()
            del rho_second

            gc.collect()
            t.cuda.empty_cache()
            errors.append(float(error))
            print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break
        
        tensor_full = t.exp(latent_1[:, None, None]) * t.exp(latent_2[None, :, None]) * t.exp(latent_3[None, None, :])

        gc.collect()
        t.cuda.empty_cache()

        return tensor_full, errors

    def prepare_2D_data(self):
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

        d1, d2 = self.tensor.shape
        # Create a mask of non-zero elements
        self.rho_sign = (self.tensor != 0)*1
            
        # Get the number of nonzeros inside each row
        self.sigma_first = self.rho_sign.sum(1)
        self.sigma_second = self.rho_sign.sum(0)

        # Take log spaceof tensor
        self.tensor_log = t.log(self.tensor)
        
        # After log, all 0 values will be -inf, so we set them to 0
        self.tensor_log[self.tensor_log == - float("Inf")] = 0.0
    
        # Initiate lantent variables
        self.latent_1 = t.zeros(d1).to(self.device)
        self.latent_2 = t.zeros(d2).to(self.device)
        
    def LLI_2D(self):
        error = 0.0

        self.rho_first = - t.div(self.tensor_log.sum(1), self.sigma_first).nan_to_num(0.0) # d1
        self.tensor_log += self.rho_first[:, None] * self.rho_sign
        self.latent_1 -= self.rho_first
        error += (self.rho_first**2).sum()


        self.rho_second = - t.div(self.tensor_log.sum(0), self.sigma_second).nan_to_num(0.0) # d2
        self.tensor_log += self.rho_second[None, :] * self.rho_sign
        self.latent_2 -= self.rho_second
        error += (self.rho_second**2).sum()

        error = float(error)
        # return the latent variables and errors
        tensor_full = t.exp(self.latent_1[:, None]) * t.exp(self.tensor_log) * t.exp(self.latent_2[None, :])
        return tensor_full, error
        
