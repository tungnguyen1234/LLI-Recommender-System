#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import torch as t


class MatrixLLI():
    def __init__(self, device, matrix, epsilon):
        self.device = device
        self.matrix = matrix
        self.epsilon = epsilon 

    def LLI(self):
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

        d1, d2 = self.matrix.shape

        # Create a mask of non-zero elements
        rho_sign = (self.matrix != 0)*1
            
        # Get the number of nonzeros inside each row
        sigma_first = rho_sign.sum(1)
        sigma_second = rho_sign.sum(0)

        # Take log spaceof matrix
        matrix_log = t.log(self.matrix)
        
        # After log, all 0 values will be -inf, so we set them to 0
        matrix_log[matrix_log == - float("Inf")] = 0.0
    
        # Initiate lantent variables
        latent_first = t.zeros((d1, 1)).to(self.device)
        latent_second = t.zeros((d2, 1)).to(self.device)
        
        # Starting the iterative steps
        step = 1

        # Iteration errors
        errors = []

        print('Start the LLI process:')
        step = 1
        while True:
            
            error = 0.0

            rho_first = - t.div(matrix_log.sum(1), sigma_first).nan_to_num(0.0) # d1
            matrix_log = rho_first[:, None] * rho_sign
            latent_first += rho_first
            error += float(rho_first**2).sum()


            rho_second = - t.div(matrix_log.sum(0), sigma_second).nan_to_num(0.0) # d1
            matrix_log = rho_second[None, :] * rho_sign
            latent_second += rho_second
            error += float(rho_second**2).sum()

            errors.append(float(error))
                
            print(f'This is step {step} with error {float(error)}')
            step += 1
            if error < self.epsilon:
                break

        # return the latent variables and errors
        return t.exp(latent_first), t.exp(latent_second), errors


