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
        self.tensor_full = None
        # Run the latent scaling

        for epoch in range(epochs):
            self.tensor_full, error = LLIProcess.LLI()
            print(f'This is step {epoch + 1} with error {float(error)}')
            # release memory
            gc.collect()
            t.cuda.empty_cache()

            self.length = None
            
            print("Here we obtain the testing values:")
            if self.dim == 2:
                # Get RMSE and MSE
                self.length = t.sum(mask_test)
                self.pred = self.tensor_full*mask_test
            elif self.dim == 3:
                # Get the testing result by getting the maximum value at the second dimension
                # get the mask of only the entries exists for the test
                mask_test_2d = t.amax(mask_test, dim = 2)
                # get the tensor by testing dim
                self.pred = t.amax(mask_test * self.tensor_full, dim = 2)
                # total test values
                self.length = t.sum(mask_test_2d)
                self.pred *= mask_test_2d
                self.original *= mask_test_2d
            
            MAE = self.mae(self.pred, self.original* mask_test)
            RMSE = self.rmse(self.pred, self.original* mask_test)
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
        
        print(self.tensor_full)
        self.topN(self.tensor_full, self.original, mask_test, N = 10)
        self.meanMAE, self.stdMAE =  np.mean(MAEs), np.std(MAEs)
        self.meanRMSE, self.stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
        # meanFCP, stdFCP =  np.mean(FCPs), np.std(FCPs)
        print(f"The overall result after {epochs} iterations")
        print(f"MAE has mean {self.meanMAE} and std {self.stdMAE}")
        print(f"RMSE has mean {self.meanRMSE} and std {self.stdRMSE}")
        # print("Precision is {:.5f} and recall is {:.5f}".format(precision, recall))
        # print("F1 score is {:.5f}".format(F1))
        # release memory
        gc.collect()
        t.cuda.empty_cache()
            
    

    def mae(self, pred, original):
        """
        Description: 
            Compute MAE (Mean Absolute Error)
        Returns: 
            MAE between prediction and original value
        """
        MAE = t.abs(pred - original).sum()/self.length
        return MAE

    def rmse(self, pred, original):
        """
        Description: 
            Compute RMSE (Root Mean Square Error)
        Returns: 
            RMSE between prediction and original value
        """
        RMSE = t.sqrt(((pred - original)**2).sum()/self.length)
        return RMSE
        
    def topN(self, pred, original, mask_test, N):

        def nan_to_num(x):
            return t.nan_to_num(x, nan= 0.0)

        def TF_rates(binary_rating, binary_top_n_items):
            # Calculate the number of true positives, false positives, true negatives, and false negatives
            TP = t.sum(binary_rating * binary_top_n_items, dim=1)
            FP = t.sum(binary_top_n_items, dim=1) - TP
            TN = t.sum((binary_rating == 0) * (binary_top_n_items == 0), dim=1)
            FN = t.sum((binary_rating == 1) * (binary_top_n_items == 0), dim=1)
            # group all terms together
            terms = [TP, FP, TN, FN]

            # remove nan elements
            for i in range(4):
                terms[i] = nan_to_num(terms[i])

            return terms
        
        def micro_F1(terms):
            TP, FP, TN, FN = terms
            # calculate precision and recall for each user
            precision = t.div(TP, TP + FP)
            recall = t.div(TP, TP + FN)
            # calculate micro by average
            mi_precision = t.mean(precision)
            mi_recall = t.mean(recall)
            # get F1 score and print
            mi_F1 = 2 * (mi_precision * mi_recall) / (mi_precision + mi_recall)
            print("Micro precision, recall, and F1 score is \
                    {:.5f}, {:.5f}, {:.5f}".format(mi_precision, mi_recall, mi_F1))
            return mi_F1

        def macro_F1(terms):
            TP, FP, TN, FN = terms
            # calculate precision and recall as total sum of each terms
            ma_precision = t.div(t.sum(TP), t.sum(TP + FP))
            ma_recall = t.div(t.sum(TP), t.sum(TP + FN))
            # get F1 score and print
            ma_F1 = 2 * (ma_precision * ma_recall) / (ma_precision + ma_recall)
            print("Macro precision, recall, and F1 score is \
                    {:.5f}, {:.5f}, {:.5f}".format(ma_precision, ma_recall, ma_F1))
            return ma_F1

        def run_all():
            users = t.unique(t.nonzero(mask_test, as_tuple = True)[0])
            num_users = len(users)
            # Define the threshold
            threshold = 0

            # Create a binary rating matrix
            binary_rating = original > threshold

            # match test user ratings
            user_rating = pred[users, :]

            # Select the top N items to recommend
            top_n_indices = t.argsort(user_rating, dim = 1, descending=True)[:, :N]

            # Create a binary matrix of top_n_items
            binary_top_n_items = t.zeros_like(original[users, :])

            # change the shape into [:, None] or view(-1, 1)
            binary_top_n_items[t.arange(num_users)[:, None], top_n_indices] = 1

            # Get all scores
            terms = TF_rates(binary_rating[users, :], binary_top_n_items)
            mi_F1 = micro_F1(terms)
            ma_F1 = macro_F1(terms)
            return mi_F1, ma_F1

        print("Top N for test matrix")
        run_all()

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


