import random
import numpy as np
from surprise import Dataset, accuracy, Reader
from surprise.model_selection import train_test_split, KFold
from surprise import SVD, NMF, SlopeOne, NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline

import os

# Default for KNN is 40.
def matrix_other_methods(percent, dataname, other_method):
    # Load the dataset with dataname
    matrix_rating = load_dataset(dataname)
    hash = {}
    kf = KFold(n_splits=3)

    MAEs = []
    RMSEs = []
    
    # Setup the algorithm object
    algo = run_algo(other_method)
    hash[other_method] = {}

    # Run the other method for steps times.
    step = 0
    print("Here is the result for", other_method, "method")
    for trainset, testset in kf.split(matrix_rating):
        print("---------------------------------")

        print(f"Step {step+1}:")
        # Fit and get predictions
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Then compute RMSE
        MAE, RMSE = accuracy.mae(predictions), accuracy.rmse(predictions)
        MAEs.append(MAE)
        RMSEs.append(RMSE)
        step += 1

    
    print("---------------------------------")
    print("The overall result is the following:")
    meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
    meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
    print(f"MAE of {other_method} has mean {meanMAE} and std {stdMAE}")
    print(f"RMSE of {other_method} has mean {meanRMSE} and std {stdRMSE}")
    print("---------------------------------")
    hash[other_method]['MAE'] = [meanMAE, stdMAE]
    hash[other_method]['RMSE'] = [meanRMSE, stdRMSE]

    print(hash)


def load_dataset(dataname):
    matrix_rating = None

    if dataname == 'ml-1m':
        # Load the movielens-1M dataset 
        file_path = os.path.expanduser('~/LLI/data/ratings.csv')
        reader = Reader(line_format=u'user item rating', sep=',', rating_scale=(1, 6), skip_lines=1)
        matrix_rating = Dataset.load_from_file(file_path, reader=reader)
    elif dataname == 'jester':
        file_path = os.path.expanduser('~/LLI/data/jester.dat')
        reader = Reader('jester')
        matrix_rating = Dataset.load_from_file(file_path, reader=reader)
    
    return matrix_rating


def run_algo(method: str):
    if method == 'svd':
        return SVD()
    if method == 'slopeone':
        return SlopeOne()
    if method == 'nmf':
        return NMF()
    if method == 'mormpred':
        return NormalPredictor()
    if method == 'knn':
        return KNNBasic()
    if method == 'knnmean':
        return KNNWithMeans()
    if method == 'knnzscore':
        return KNNWithZScore()
    if method == 'knnbaseline':
        return KNNBaseline()

