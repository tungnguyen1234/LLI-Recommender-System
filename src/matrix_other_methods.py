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
    
    steps = 3

    prt = 'nmf' 
    other_methods = ('svd', 'slopeone', 'mormpred', 'knn', 'knnmean', 'knnzscore', 'knnbaseline')
    hash = {}
    kf = KFold(n_splits=3)

    for other_method in other_methods:
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

            print("Step " +str(step+1))
            # Fit and get predictions
            algo.fit(trainset)
            predictions = algo.test(testset)

            # Then compute RMSE
            MAE, RMSE = accuracy.mae(predictions), accuracy.rmse(predictions)
            MAEs.append(MAE)
            RMSEs.append(RMSE)
            step += 1

        hash[other_method]['MAE'] = MAEs
        hash[other_method]['RMSE'] = RMSEs
        print("---------------------------------")
        print("The overall result is the following:")
        print("MAE after", steps,"steps is", MAEs)
        print("RMSE after", steps,"steps is", RMSEs)
        print("---------------------------------")

    # ml-1m
    # {"svd": {'MAE': [0.6970922020333427, 0.6964908028564362, 0.697628807736389], 'RMSE': [0.8861347590600661, 0.8850750975897014, 0.8866755258325019]},  / 
    # "slopeone": {'MAE': [0.7162708707189671, 0.7163285019240915, 0.7165699595026147], 'RMSE': [0.9086292503350797, 0.9084501605019583, 0.9082617144724582]}, /
    #  'mormpred': {'MAE': [1.2572714798621918, 1.2555584527777122, 1.2562090359490639], 'RMSE': [1.5647396673893257, 1.5616273631714392, 1.5611985171541871]}, / 
    # 'knn': {'MAE': [0.7343407906642845, 0.7342878157616762, 0.7349294824128495], 'RMSE': [0.9311228520456881, 0.9302181905834354, 0.9311455466928018]}, / 
    # 'knnmean': {'MAE': [0.7428340993001977, 0.7396609322067648, 0.7438968359953264], 'RMSE': [0.934147499849567, 0.930057697900876, 0.9361619235149201]}, /
    # 'knnzscore': {'MAE': [0.7384678544251281, 0.7392445598486074, 0.7396291049724848], 'RMSE': [0.9334756403390017, 0.9336496545832345, 0.9347959415340819]},/
    # 'knnbaseline': {'MAE': [0.7087580235850927, 0.7114469103614695, 0.7116378061048275], 'RMSE': [0.8984211459910295, 0.9005428451333046, 0.9004954417773051]}}

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

