#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import numpy as np, gc, torch
from TensorScore import TensorScore
from TensorObject import TensorObject


class Tensor(TensorObject):
    def __init__(self, device, dim, dataname, num_feature, percent, epsilon, steps, limit):
        super().__init__(device, dim, dataname, percent, limit)
        '''
        Desciption:
            This function runs all the steps to pre-processing MovieLens data, running the tensor latent
            algorithm, and retrieving the MAE and RMSE score. 
        Input:
            device: 
                The gpu device to run the algorithm on.
            percent: float
                The percentage of splitting for training and testing data. Default is None.
            dataname: str
                The dataset to run the algorithm on.
            limit: int 
                The limit number of data that would be process. Default is None, meaning having no limit
            age, occup, gender: str
                The features by string that would be added in the third dimension. There are three types:
                age, occupation, and gender.
            percent: str
                The percentage of splitting for training and testing
            epsilon: float
                The convergence threshold for the algorithm.
        Output:
            Prints the MAE, RMSE and errors from the latent scaling convergence steps.
        '''
        self.num_feature = num_feature
        self.epsilon = epsilon
        self.steps = steps
        self.features = self.get_features_by_num()

        if not (0<= self.percent <1):
            self.percent = 1

    
    def performance_overal_LLI(self):
        if self.dim == 2:
            self.performance(feature = None)
            # release memory
            gc.collect()
            torch.cuda.empty_cache()

        elif self.dim == 3:
            assert self.dataname == 'ml-1m'
            for feature in self.features:
                self.performance(feature)

            # release memory
            gc.collect()
            torch.cuda.empty_cache()

    def get_features_by_num(self):
        # this method only works for tensor larger than 3 dim
        if self.num_feature == 1:
            return [{"occup"}, {"age"}, {"gender"}]
        if self.num_feature == 2:
            return [{"age", "occup"}, {"occup", "gender"}, {"age", "gender"}]
        if self.num_feature == 3:
            return [{"age", "occup", "gender"}]


    def performance(self, feature):
        '''
        Desciption:
            This function runs all the steps to pre-processing MovieLens data, running the tensor latent
            algorithm, and retrieving the MAE and RMSE score. 
        Input:
            steps: the number of steps to run the algorithm
        Output:
            Prints the MAE, RMSE and errors from the latent scaling convergence steps.
        '''
        MAEs = RMSEs = FCPs = []
        list_errors = []
        
        output_text = f"result/LLI_{self.dataname}_dim_{self.dim}_{self.num_feature}.txt"
        # os.remove(output_text)

        self.tensor_score = TensorScore(self.device, self.dim, feature,\
                                        self.dataname, self.percent, self.epsilon, self.limit)

        print("-------------------------------------------------")
        print(f"Here we test the algorithm with feature {feature}")
        print(f"The algorithm runs {self.steps} times to get the mean and std!")
        for i in range(self.steps):
            print(f"Step {i+1}:")
            self.tensor_score.tensor_pred()
            MAE = self.tensor_score.mae()
            RMSE = self.tensor_score.rmse()
            FCP = self.tensor_score.fcp()
            MAEs.append(float(MAE))
            RMSEs.append(float(RMSE))
            FCPs.append(float(FCP))
            list_errors.append(str(self.tensor_score.errors))
            print(f"MAE is {float(MAE)}")
            print(f"RMSE is {float(RMSE)}")
            print(f"FCP is {float(FCP)}")
            print("-------------")
            # release memory
            gc.collect()
            torch.cuda.empty_cache()


        meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
        meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
        meanFCP, stdFCP =  np.mean(FCPs), np.std(FCPs)
        print(f"The overall result after {self.steps} steps")
        print(f"MAE has mean {meanMAE} and std {stdMAE}")
        print(f"RMSE has mean {meanRMSE} and std {stdRMSE}")
        print(f"FCP has mean {meanFCP} and std {stdFCP}")

        
        
        lines = [f"Here we test the algorithm with feature {feature}",\
                "-------------------------------------------------",\
                f"MAE has mean {meanMAE} and std {stdMAE}", \
                f"RMSE has mean {meanRMSE} and std {stdRMSE}",\
                f"FCP has mean {meanFCP} and std {stdFCP}",\
                f"The errors after {self.steps} steps are:"]\
                + list_errors \
                + ["-------------------------------------------------", "\n\n"]
        with open(output_text, "a", encoding='utf-8') as f:
            f.write('\n'.join(lines))


    


