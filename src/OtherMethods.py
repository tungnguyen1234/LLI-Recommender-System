import numpy as np
from surprise import Dataset, accuracy
from surprise.model_selection import KFold
from surprise import SVD, NMF, SlopeOne, NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, CoClustering, SVDpp


def OtherMethods(percent, dataname, other_method):
    n_splits = int(1/percent)

    # Load the dataset with dataname
    
    output_text = f"result/{dataname}_2_dim_methods.txt"

    matrix_rating = load_dataset(dataname)
    kf = KFold(n_splits=n_splits)
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
    matrix_rating = Dataset.load_builtin(dataname)
    return matrix_rating


def run_algo(method: str):
    sim_options = {'name':'cosine', 'user_based':False}
    if method == 'svd':
        return SVD()
    if method == 'svdpp':
        return SVDpp()
    if method == 'slope_one':
        return SlopeOne()
    if method == 'nmf':
        return NMF(biased = True)
    if method == 'norm_pred':
        return NormalPredictor()
    if method == 'co_clustering':
        return CoClustering()
    if method == 'knn_basic':
        return KNNBasic(k=25, min_k=5, sim_options=sim_options, verbose=True)
    if method == 'knn_with_means':
        return KNNWithMeans(k=25, min_k=5, sim_options=sim_options, verbose=True)
    if method == 'knn_with_z_score':
        return KNNWithZScore(k=25, min_k=5, sim_options=sim_options, verbose=True)
    if method == 'knn_baseline':
        return KNNBaseline(k=25, min_k=5, sim_options=sim_options, verbose=True)
    

