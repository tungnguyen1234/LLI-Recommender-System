#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import pandas as pd
import torch as t
import numpy as np
from matrix_movieLens import matrix_construct
from TensorScore import TensorScore
from TensorData import TensorData



class Tensor():
    def __init__(self, device, dataname, age, occup, gender, percent, epsilon, limit = None):
        self.device = device 
        self.percent = percent
        self.limit = limit 
        self.dataname = dataname
        self.epsilon = epsilon

        
        self.tensor_data = TensorData(self.device, self.dataname, self.limit)
        self.ages, self.occupations, self.genders = self.tensor_data.extract_features()
        

        self.features= set()
        if age == 'True':
            self.features.add("age")
        if occup == 'True':
            self.features.add("occup")
        if gender == 'True':
            self.features.add("gender")

        self.matrix_rating = matrix_construct(self.device)
        self.tensor_score = TensorScore(self.device, self.matrix_rating, self.features, self.ages, \
                                        self.occupations, self.genders, self.percent, self.epsilon,)


        if not (0<= self.percent <1):
            self.percent = 1


    def retrieve_result(self, steps = 2):
        '''
        Desciption:
            This function runs all the steps to pre-processing MovieLens data, running the tensor latent
            algorithm, and retrieving the MAE and RMSE score. 
        Input:
            percent: int
                The percentage of splitting for training and testing data. Default is None.
            limit: int 
                The limit number of data that would be process. Default is None, meaning having no limit
            features: set()
                The features by string that would be added in the third dimension. There are three types:
                age, occupation, and gender.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Prints the MAE, RMSE and errors from the latent scaling convergence steps.
        '''
        MAEs = []
        RMSEs = []
        list_errors = []
        
        output_text = f"result/tensor_{self.dataname}.txt"
        # os.remove(output_text)

        

        print("The algorithm runs 2 times to get the mean and std!")
        for i in range(steps):
            print("-------------------------------------------------")
            print(f"Step {i+1}:")
            MAE, RMSE, errors = self.tensor_score.tensor_score()
            MAE = float(MAE)
            RMSE = float(RMSE)
            MAEs.append(MAE)
            RMSEs.append(RMSE)
            list_errors.append(errors)
        print("-------------------------------------------------")    
        mean_errors = [np.mean([list_errors[0][i], list_errors[-1][i]]) for i in range(len(list_errors[0]))]
        std_errors = [np.std([list_errors[0][i], list_errors[-1][i]]) for i in range(len(list_errors[0]))]
        meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
        meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
        print(f"MAE has mean {meanMAE} and std {stdMAE}")
        print(f"RMSE has mean {meanRMSE} and std {stdRMSE}")


        
        lines = [f"MAE has mean {meanMAE} and std {stdMAE}", f"RMSE has mean {meanRMSE} and std {stdRMSE}",\
                f"Error means from the iteration process is {mean_errors}", \
                f"Error stds from the iteration process is {std_errors}",]
        with open(output_text, "a", encoding='utf-8') as f:
            f.write('\n'.join(lines))

        

        # Testing purpose
        # matrix_rating = t.tensor([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
        # ages = t.tensor([1, 20, 30])
        # occupations = t.tensor([0, 4, 5])
        # genders = t.tensor([0, 1, 0])


    


