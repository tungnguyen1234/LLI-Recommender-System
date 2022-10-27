import torch as t, numpy as np
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

    def tensor_pred(self, epochs): 
        '''
        Desciption:
            This function splits a training tensor for latent scaling algorithm. For testing, we obtain the 
            recommendation result as the maximum value by the feature dimension for each (user, product) pair 
            as max(tensor[user, product, :]) 
        Output:
            Returns the original matrix and prediction result from latent scaling convergence steps.
        '''
        self.original, tensor_train, mask_test = TrainTest(self.device, self.dim, self.feature, \
                                    self.dataname, self.percent, self.limit).train_test()

        LLIProcess = TensorLLI(self.device, self.dim, tensor_train, self.epsilon)
        LLIProcess.prepare_2D_data()
        MAEs, RMSEs = [], []
        self.list_errors = []
        # Run the latent scaling
        for epoch in range(epochs):
            self.pred, error = LLIProcess.LLI(epoch)
            print(f'This is step {epoch + 1} with error {float(error)}')
            # release memory
            gc.collect()
            t.cuda.empty_cache()

            tensor_test = None
            self.length = None
            
            print("Here we obtain the testing values:")
            if self.dim == 2:
                # Get RMSE and MSE
                self.length = t.sum(mask_test)
                self.pred *= mask_test
                self.original *= mask_test
            elif self.dim == 3:
                # Get the testing result by getting the maximum value at the second dimension
                # get the mask of only the entries exists for the test
                mask_test_2d = t.amax(mask_test, dim = 2)
                # get the tensor by testing dim
                self.pred = t.amax(mask_test * self.pred, dim = 2)
                # total test values
                self.length = t.sum(mask_test_2d)
                self.pred *= mask_test_2d
                self.original *= mask_test_2d

            
            MAE = self.mae()
            RMSE = self.rmse()
            # FCP = self.tensor_score.fcp()
            MAEs.append(float(MAE))
            RMSEs.append(float(RMSE))
            # FCPs.append(float(FCP))
            self.list_errors.append(str(error))
            print(f"MAE is {float(MAE)}")
            print(f"RMSE is {float(RMSE)}")
            # print(f"FCP is {float(FCP)}")
            print("-------------")
            # release memory
            gc.collect()
            t.cuda.empty_cache()
        

        self.meanMAE, self.stdMAE =  np.mean(MAEs), np.std(MAEs)
        self.meanRMSE, self.stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
        # meanFCP, stdFCP =  np.mean(FCPs), np.std(FCPs)
        print(f"The overall result after {epochs} iterations")
        print(f"MAE has mean {self.meanMAE} and std {self.stdMAE}")
        print(f"RMSE has mean {self.meanRMSE} and std {self.stdRMSE}")

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
        MAE = t.abs(self.pred - self.original).sum()/self.length
        return MAE

    def rmse(self):
        """
        Description: 
            Compute RMSE (Root Mean Square Error)
        Returns: 
            RMSE between prediction and original value
        """
        RMSE = t.sqrt(((self.pred - self.original)**2).sum()/self.length)
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
            predictions_u[int(u)].append((self.original[int(u), int(p)], self.pred[int(u), int(p)]))

        for u0, preds in tqdm(predictions_u.items()):
            for r0i, esti in preds:
                for r0j, estj in preds:
                    if esti > estj and r0i > r0j:
                        nc_u[u0] += 1
                    if esti >= estj and r0i < r0j:
                        nd_u[u0] += 1

        nc = t.mean(t.tensor(list(nc_u.values()), dtype = float))
        nd = t.mean(t.tensor(list(nd_u.values()), dtype = float))

        try:
            fcp = nc / (nc + nd)
        except ZeroDivisionError:
            raise ValueError('cannot compute fcp on this list of prediction. ' +
                            'Does every user have at least two predictions?')

        return fcp


