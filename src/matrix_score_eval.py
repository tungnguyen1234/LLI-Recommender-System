
import torch as t
import numpy as np
from math import *
from matrix_latent import matrix_latent

def matrix_traintest_score(matrix, percent, epsilon):
    '''
    Desciption:
        This function splits a training matrix for latent scaling algorithm and a testing vector to compare 
        and retrivie 
    Input:
        matrix: torch.tensor 
            The matrix of user rating on products.
        percent: int
            The percentage of splitting for training and testing data. This value ranges from 
            0 to 1 and default is None.
        epsilon: float
            The convergence number for the algorithm.
    Output:
        Returns the MAE, RMSE and errors from the latent scaling convergence steps.
    '''

    if not (0<= percent <1):
        percent = 1

    user_product = t.nonzero(matrix)
    per = t.randperm(len(user_product))
    num_test = int(percent*len(user_product))
    test = per[:num_test]

    re_train = []
    re_test = []

    # Setup
    for i in range(len(test)):
        user, product = user_product[test[i]]
        rating = matrix[user, product].clone()
        re_train.append(rating)
        matrix[user, product] = 0

    # Scaling
    latent_user, latent_prod, errors = matrix_latent(matrix, epsilon)
    # Test
    MAE = 0.0
    MSE = 0.0
    for i in range(len(test)):
        user, product = user_product[test[i]]
        rating = 1/(latent_user[user]*latent_prod[product])
        diff = float(abs(re_train[i] - rating))
        re_test.append(rating)
        MAE += diff
        MSE += diff**2

    MAE = MAE/len(test)
    RMSE = np.sqrt(MSE/len(test))
    return MAE, RMSE, errors
