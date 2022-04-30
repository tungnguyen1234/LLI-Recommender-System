from tensor_3D_latent import tensor_latent
import numpy as np
import torch as t


def tensor_traintest_score(tensor, percent, epsilon): 
    '''
    Desciption:
        This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
        recommendation result as the maximum value by the feature dimension for each (user, product) pair 
        as max(tensor[user, product, :]) 
    Input:
        tensor: t.tensor 
            The tensor of user ratings on products based on different features
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

    user_prod_feat = t.nonzero(tensor)
    per = t.randperm(len(user_prod_feat))
    # Get random test by percent
    num_test = int(percent*len(user_prod_feat))
    test = per[:num_test]

    re_train = {}
    re_test = {}

    # Setup
    for i in range(len(test)):
        user, product, feature = user_prod_feat[test[i]]
        rating = tensor[user, product, feature].clone()
        re_train[(int(user), int(product))] = rating
        tensor[user, product, feature] = 0

    # Run the latent scaling
    latent_user, latent_prod, latent_feature, errors = tensor_latent(tensor, epsilon)

    # Test
    MAE = 0.0
    MSE = 0.0
    for i in range(len(test)):
        user, product, feature = user_prod_feat[test[i]]
        rating = 1/(latent_user[user]*latent_prod[product]*latent_feature[feature])
        comp = re_train[(int(user), int(product))]
        re_test[int(user), int(product)] = t.max(comp, rating)
        
    for key, rating in re_test.items():
        diff = float(abs(re_train[key] - re_test[key]))
        MAE += diff
        MSE += diff**2

    MAE = MAE/len(test)
    RMSE = np.sqrt(MSE/len(test))
    return MAE, RMSE, errors

