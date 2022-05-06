import torch as t
from TensorTrainTest import TrainTest
from TensorLLI import TensorLLI
from tqdm import tqdm 

class TensorScore(): 
    def __init__(self, device, matrix, features, ages, occupations, genders, percent, epsilon):
        self.percent = percent
        self.epsilon = epsilon
        self.device = device
        self.matrix = matrix
        self.features = features
        self.ages = ages
        self.occupations = occupations
        self.genders = genders

        self.train_test = TrainTest(self.device, self.matrix, self.features, \
            self.ages, self.occupations, self.genders, self.percent)

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

        # Run the latent scaling
        tensor_train, mask_test = self.train_test.train_test()
        self.tensor_LLI = TensorLLI(self.device, tensor_train, self.epsilon)
        tensor_full, errors = self.tensor_LLI.LLI()

        # Get the testing result by getting the maximum value at the second dimension
        matrix_test = t.amax(mask_test * tensor_full + tensor_train, dim = 2)
        print("Here we obtain the testing values:")

        # Get RMSE and MSE
        mae_loss = t.nn.L1Loss()
        mse_loss = t.nn.MSELoss()
        RMSE = t.sqrt(mse_loss(matrix_test, self.matrix))
        MAE = mae_loss(matrix_test, self.matrix)

        return MAE, RMSE, errors


