from sklearn.model_selection import GridSearchCV
from TensorPMF import PMF_GS
from CheckPMF import *
import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataname", choices=('ml-100k', 'ml-1m', 'jester', 'ml-10m'), default='ml-100k')
    parser.add_argument("--num_bt", type = int, required=False, default=10)
    parser.add_argument("--size_bt", type = int, required=False, default=1000)
    parser.add_argument("--percent", type=float, required=False, default = 0.2)
    parser.add_argument("--chunksize", type=float, required=False, default = 6)
    args = parser.parse_args()

    ratings = load_rating_data(args.dataname)
    # limit ratings to 1000
    # limit = 10
    # ratings = ratings[:limit]

    num_user = int(np.amax(ratings[:, 0])) + 1  # user总数
    num_item = int(np.amax(ratings[:, 1])) + 1  # movie总数
    print(num_user, num_item)

    # Define the hyperparameters to search

    list_check = [0.01, 0.05, 0.1, 0.5, 1]

    param_grid = {'num_feat': list(range(5, 16, 5)), 'gamma': list_check, 
                '_lambda_U': list_check, '_lambda_P': list_check, 'momentum': list_check}

    # Create a model to search
    model = PMF_GS()

    ratings = load_rating_data(args.dataname)
    ratings_chunks = data_chunks(ratings, chunks = args.chunksize)

    # Create a grid search object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid)

    # Fit the grid search to the data
    X, y = ratings_chunks[0][:, :2], ratings_chunks[0][:, 2]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.percent)  
    grid_search.fit(X, y, num_user = num_user, num_item = num_item)

    # Print the best parameters
    print(grid_search.best_params_)

    new_params = grid_search.best_params_
    # new_params = {'_lambda_P': 0.01, '_lambda_U': 0.01, 'gamma': 1, 
    #                 'momentum': 1, 'num_feat': 15, 'num_item': 3953, 'num_user': 6041}
    new_params['num_batches'] = args.num_bt
    new_params['batch_size'] = args.size_bt
    
    PMF_visual_chunks(ratings_chunks, args.dataname, **new_params)