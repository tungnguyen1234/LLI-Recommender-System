import torch as t
from TensorTrainTest import TrainTest
from TensorLLI import TensorLLI
from tqdm import tqdm 

class TensorScore(): 
    def __init__(self, device, matrix_rating, features, ages, occupations, genders, percent, epsilon):
        self.percent = percent
        self.epsilon = epsilon
        self.device = device
        self.matrix_rating = matrix_rating
        self.features = features
        self.ages = ages
        self.occupations = occupations
        self.genders = genders

        self.train_test = TrainTest(self.device, self.matrix_rating, self.features, \
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
        tensor_train, test_idx = self.train_test.train_test()
        self.tensor_LLI = TensorLLI(self.device, tensor_train, self.epsilon)
        latent_user, latent_prod, latent_feature, errors = self.tensor_LLI.LLI()

        print("Here we obtain the testing values")
        re_test = {}
        # Get the maximum rating for each user and product 
        for user, product, feature in tqdm(test_idx):
            rating = 1/(latent_user[user]*latent_prod[product]*latent_feature[feature])
            val = self.matrix_rating[int(user), int(product)]
            re_test[(user, product)] = t.max(val, rating)
            
        
        # Regroup the ratings to get RMSE and MSE
        score_train = [] 
        score_test = []
        for key, rating in tqdm(re_test.items()):
            user, product = key
            score_train.append(self.matrix_rating[int(user), int(product)])
            score_test.append(rating)

        # Get RMSE and MSE
        mae_loss = t.nn.L1Loss()
        mse_loss = t.nn.MSELoss()
        score_train, score_test = t.tensor(score_train), t.tensor(score_test)
        RMSE = t.sqrt(mse_loss(score_train, score_test))
        MAE = mae_loss(score_train, score_test)

        return MAE, RMSE, errors


