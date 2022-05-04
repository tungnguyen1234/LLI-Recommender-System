import random
import numpy as np
from surprise import Dataset, accuracy, Reader
from surprise.model_selection import train_test_split, KFold
from surprise import SVD, NMF, SlopeOne, NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline

import os

# Default for KNN is 40.
def matrix_other_methods(percent, dataname, other_method):
    # Load the dataset with dataname
    
    path = "result/"
    output_text = path + str(dataname) + ".txt"

    matrix_rating = load_dataset(dataname)
    kf = KFold(n_splits=3)
    MAEs = []
    RMSEs = []
    
    # Setup the algorithm object
    algo = run_algo(other_method)

    # Run the other method for steps times.
    step = 0
    print(f"Here is the result of dataset {dataname} for {other_method} method")
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

    
    print("-------------------------------------------------")   
    meanMAE, stdMAE =  np.mean(MAEs), np.std(MAEs)
    meanRMSE, stdRMSE =  np.mean(RMSEs), np.std(RMSEs)
    print(f"MAE has mean {meanMAE} and std {stdMAE}")
    print(f"RMSE has mean {meanRMSE} and std {stdRMSE}")

    lines = [f"Here is the result of dataset {dataname} for {other_method} method",\
            "---------------------------------", \
            f"MAE has mean {meanMAE} and std {stdMAE}",\
            f"RMSE has mean {meanRMSE} and std {stdRMSE}", \
            "---------------------------------", "\n"]
    with open(output_text, "a", encoding='utf-8') as f:
        f.write('\n'.join(lines))

def load_dataset(dataname):
    matrix_rating = None

    if dataname == 'ml-1m':
        # Load the movielens-1M dataset 
        file_path = os.path.expanduser('~/LLI/data/ratings.csv')
        reader = Reader(line_format=u'user item rating', sep=',', rating_scale=(1, 6), skip_lines=1)
        matrix_rating = Dataset.load_from_file(file_path, reader=reader)
    elif dataname == 'jester':
        # file_path = os.path.expanduser('jester_rating.dat')
        # file_path = os.path.expanduser('~/LLI/data/jester.dat')
        # reader = Reader(line_format='user item rating ', sep=',',rating_scale=(1,20))
        matrix_rating = Dataset.load_builtin('jester')

    return matrix_rating


def run_algo(method: str):
    sim_options={'name':'pearson','min_support':5,'user_based':True}
    if method == 'svd':
        return SVD()
    if method == 'slopeone':
        return SlopeOne()
    if method == 'nmf':
        return NMF(biased = True)
    if method == 'normpred':
        return NormalPredictor()
    if method == 'knn':
        return KNNBasic(k=25,min_k=5,sim_options=sim_options,verbose=True)
    if method == 'knnmean':
        return KNNWithMeans(k=25,min_k=5,sim_options=sim_options,verbose=True)
    if method == 'knnzscore':
        return KNNWithZScore(k=25,min_k=5,sim_options=sim_options,verbose=True)
    if method == 'knnbaseline':
        return KNNBaseline(k=25,min_k=5,sim_options=sim_options,verbose=True)

