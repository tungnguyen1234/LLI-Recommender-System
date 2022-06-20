import torch as t
from tqdm import tqdm
from TensorTrainTest import TrainTest
from TensorLLI import TensorLLI
from TensorObject import TensorObject
import gc
from collections import defaultdict


class TensorScore(TensorObject): 
    def __init__(self, device, dim, feature, dataname, percent, epsilon, limit): 
        super().__init__(device, dim, dataname, percent, limit)
        self.epsilon = epsilon
        self.feature = feature

    def tensor_pred(self): 
        '''
        Desciption:
            This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
            recommendation result as the maximum value by the feature dimension for each (user, product) pair 
            as max(tensor[user, product, :]) 
        Output:
            Returns the original matrix and prediction result from latent scaling convergence steps.
        '''
        self.train_test = TrainTest(self.device, self.dim, self.feature, 
                                    self.dataname, self.percent, self.limit)
        # Run the latent scaling
        tensor_2_dim, tensor_train, mask_test = self.train_test.train_test()

        tensor_LLI = TensorLLI(self.device, self.dim, tensor_train, self.epsilon)
        tensor_full, errors = tensor_LLI.LLI()

        tensor_test = None
        self.pred = self.org = self.length = None
        self.errors = errors

        print("Here we obtain the testing values:")
        if self.dim == 2:
            tensor_test = tensor_full * mask_test
            # Get RMSE and MSE
            self.length = len(mask_test.nonzero())
            self.pred = tensor_test * mask_test
            self.org = tensor_2_dim * mask_test

            
        elif self.dim == 3:
            # Get the testing result by getting the maximum value at the second dimension
            mask_full = tensor_train.clone()
            mask_full = (mask_full != 0)*1 + mask_test
            tensor_test = t.amax(mask_full * tensor_full, dim = 2)
            self.length = len(tensor_2_dim.nonzero())
            self.pred = tensor_test
            self.org = tensor_2_dim

        # release memory
        gc.collect()
        t.cuda.empty_cache()
        
    

    def mae(self):
        """
        Description: 
            Compute MAE (Mean Absolute Error)
        Returns: 
            MAE between prediction and original value
        """
        pred_copy = t.flatten(self.pred.clone())
        org_copy = t.flatten(self.org.clone())
        MAE = t.abs(pred_copy - org_copy).sum()/self.length
        return MAE

    def rmse(self):
        """
        Description: 
            Compute RMSE (Root Mean Square Error)
        Returns: 
            RMSE between prediction and original value
        """
        pred_copy = t.flatten(self.pred.clone())
        org_copy = t.flatten(self.org.clone())
        RMSE = t.sqrt(((pred_copy - org_copy)**2).sum()/self.length)
        return RMSE


    def fcp(self, verbose=True):
        """
        Description: Compute FCP (Fraction of Concordant Pairs).
            Computed as described in paper `Collaborative Filtering on Ordinal User
            Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
            and Sill, section 5.2.
        Returns:
            The Fraction of Concordant Pairs.
        Raises:
            ValueError: When ``predictions`` is empty.
        """

        predictions_u = defaultdict(list)
        nc_u = defaultdict(int)
        nd_u = defaultdict(int)

        for u, p in tqdm(self.pred.nonzero()):
            predictions_u[int(u)].append((self.org[int(u), int(p)], self.pred[int(u), int(p)]))

        for u0, preds in tqdm(predictions_u.items()):
            for r0i, esti in preds:
                for r0j, estj in preds:
                    if esti > estj and r0i > r0j:
                        nc_u[u0] += 1
                    if esti >= estj and r0i < r0j:
                        nd_u[u0] += 1

        nc = t.mean(t.tensor(list(nc_u.values()), dtype = float))
        nd = t.mean(t.tensor(list(nd_u.values()), dtype = float))

        print(nc, nd)

        try:
            fcp = nc / (nc + nd)
        except ZeroDivisionError:
            raise ValueError('cannot compute fcp on this list of prediction. ' +
                            'Does every user have at least two predictions?')

        return fcp


