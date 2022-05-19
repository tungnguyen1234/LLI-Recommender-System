import torch as t
from TensorTrainTest import TrainTest
from TensorLLI import TensorLLI
from TensorObject import TensorObject
import gc

class TensorScore(TensorObject): 
    def __init__(self, device, dim, feature, dataname, percent, epsilon, limit):
        super().__init__(device, dim, dataname, percent, limit)
        self.epsilon = epsilon
        self.feature = feature

    '''
    Desciption:
        This class retrieves values for testing purposes on the LLI algorithm
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

    def tensor_score(self): 
        '''
        Desciption:
            This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
            recommendation result as the maximum value by the feature dimension for each (user, product) pair 
            as max(tensor[user, product, :]) 
        Input:
            tensor: torch.tensor 
                The tensor of user ratings on products based on different features
            percent: int
                The percentage of splitting for training and testing data. This value ranges from 
                0 to 1 and default is None.
            epsilon: float
                The convergence number for the algorithm.
        Output:
            Returns the MAE, RMSE and errors from the latent scaling convergence steps.
        '''
        self.train_test = TrainTest(self.device, self.dim, self.feature, 
                                    self.dataname, self.percent, self.limit)
        # Run the latent scaling
        tensor_2_dim, tensor_train, mask_test = self.train_test.train_test()

        tensor_LLI = TensorLLI(self.device, self.dim, tensor_train, self.epsilon)
        tensor_full, errors = tensor_LLI.LLI()

        mae_loss = t.nn.L1Loss()
        mse_loss = t.nn.MSELoss()
        tensor_test = None
        RMSE = MAE = None
        
        print("Here we obtain the testing values:")
        if self.dim == 2:
            tensor_test = tensor_full * mask_test
            # Get RMSE and MSE
            length = len(mask_test.nonzero())
            a = t.flatten(tensor_test)
            b = t.flatten(tensor_2_dim * mask_test)

            RMSE = t.sqrt(((a - b)**2).sum()/length)
            MAE = t.abs(a - b).sum()/length
        elif self.dim == 3:
            # Get the testing result by getting the maximum value at the second dimension
            tensor_test = t.amax(mask_test * tensor_full + tensor_train, dim = 2)
            length = len(tensor_2_dim.nonzero())
            a = t.flatten(tensor_test)
            b = t.flatten(tensor_2_dim)

            RMSE = t.sqrt(((a - b)**2).sum()/length)
            MAE = t.abs(a - b).sum()/length

        # release memory
        gc.collect()
        t.cuda.empty_cache()
        return MAE, RMSE, errors


