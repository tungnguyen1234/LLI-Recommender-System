import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from TensorPMF import PMF
import numpy as np
from os.path import join
from dataset import Dataset
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from scipy.stats import truncnorm


def get_truncated_normal(mean=0, sd=1, low=0, upp=10, size = 1000):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, size=size)

def generate_norm_dist(dataname):
    prefer = []
    data = Dataset.load_builtin(dataname)
    n = len(data.raw_ratings)
    norm_data = get_truncated_normal(mean=5.5, sd=1, low=1, upp=10, size=n)

    for i, line in enumerate(data.raw_ratings):  
        (userid, movieid, rating, ts) = line
        uid = int(userid)
        mid = int(movieid)
        prefer.append([uid, mid, norm_data[i]])
    data = np.array(prefer)
    return norm_data, data


def load_rating_data(dataname):
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = []
    data = Dataset.load_builtin(dataname)
    columns = ["UserID", "ProductID","Rating","Timestamp"]
    df = pd.DataFrame(data.raw_ratings, columns = columns)
    for col in columns:
        df[col] = pd.to_numeric(df[col])
    df = df[columns[:3]]
    data = df.values
    return data

def data_chunks(data, chunks = 10):
    for _ in range(chunks):
        np.random.shuffle(data)
    return np.array_split(data, chunks)

def get_plot_dir():
    """Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.data/ and path is 'PWD'
    """

    folder = os.environ.get('PWD') + '/.plot'
    if not os.path.exists(folder):
        os.makedirs(folder) 
    return folder

def get_plot(string):
    return join(get_plot_dir(), string)

def get_dir_result():
    folder = os.environ.get('PWD') + '/.result'
    if not os.path.exists(folder):
        os.makedirs(folder) 
    return folder

def get_result(string):
    return join(get_dir_result(), string)


def save_result(dataname, test_data, predict):

    print("Mean and var ground truth is %d and %d", np.mean(test_data), np.var(test_data))
    print("Mean and var prediction is %d and %d", np.mean(predict), np.var(predict))

    lines = [f"Here we test with dataset {dataname}",\
    "-------------------------------------------------",\
    f"Mean and var ground truth is {np.mean(test_data)} and {np.var(test_data)}", \
    f"Mean and var prediction is {np.mean(predict)} and {np.var(predict)}"
    + "\n", "-------------------------------------------------", "\n\n"]

    folder = get_dir_result()
    output_text = folder + f'/PMF_{dataname}.txt'
    
    with open(output_text, "a+", encoding='utf-8') as f:
        f.write('\n'.join(lines))
        f.close()

def PMF_visualizations(ratings, dataname, percent, **kwargs):

    train, test = train_test_split(ratings, test_size=percent)  
    sns.kdeplot(train[:, 2], label="Train")
    sns.kdeplot(test[:, 2], label="Test")
    path = get_plot(args.dataname + '_data_dist')
    plt.xlabel('ratings')
    plt.legend()
    plt.savefig(path)
    plt.show()

    datas = [[train, test], [test, train]]
    RMSEs = []
    for i, (tra, te) in enumerate(datas):
        pmf = PMF()
        pmf.set_params(**kwargs)
        pmf.fit(tra, te)
        # Check performance by plotting train and test errors
        plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label="Train")
        plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label="Test")
        RMSEs.append([pmf.rmse_train[-1], pmf.rmse_test[-1]])
        plt.title('The MovieLens Dataset Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        path = join(get_plot_dir(), dataname+ f'_{i+1}')
        plt.savefig(path)
        plt.show()

        predict = pmf.test_ratings(te)
        test_data = list(te[:, 2])
        save_result(dataname, test_data, predict)

        # save result
        tp = ["Predict"]*len(test_data) + ["Ground Truth"]*len(predict)
        datum = [*test_data, *predict]
        df = pd.DataFrame(data={'type': tp, 'data':datum})
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        sns.boxplot(x='data', y='type', data=df, ax=ax_box, showfliers = False, dodge=False)
        sns.kdeplot(data=predict, label="Predict", ax=ax_hist)
        sns.kdeplot(data=test_data, label="Ground Truth", ax=ax_hist)
        ax_box.set(xlabel='', ylabel='')
        ax_hist.set(xlabel='Test set ratings')
        plt.legend(labels=["Predict","Ground Truth"])
        path = get_plot(dataname + f'_test_data_{i}')
        plt.savefig(path)
        plt.legend()
        plt.show()


    print(RMSEs)
    # ml-1m: precision_acc,recall_acc:(0.055772734802054344, 0.025622311369641552)
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))


def PMF_visual_chunks(ratings, dataname, **kwargs):

    chunks = len(ratings)
    train = ratings[0]
    tests = ratings[1:]
    # breakpoint()
    pairs_test = max([test_vec.shape[0] for test_vec in tests])
    pmf = PMF()
    pmf.set_params(**kwargs)

    pmf.fit_train(train, **kwargs)
    RMSEs = []

    for i, test in enumerate(tests):
        pmf.get_RMSE_test(test, pairs_test)
        # pmf.fit(train, test)
        # Check performance by plotting train and test errors
        plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label=f"Test_{i+1}")
        RMSEs.append([pmf.rmse_train[-1], pmf.rmse_test[-1]])
        predict = pmf.test_ratings(test)
        test_data = list(test[:, 2])
        save_result(dataname, test_data, predict)
        pmf.rmse_test = []
        
        # save result
        # tp = ["Predict"]*len(test_data) + ["Ground Truth"]*len(predict)
        # datum = [*test_data, *predict]
        # df = pd.DataFrame(data={'type': tp, 'data':datum})
        # f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
        # sns.boxplot(x='data', y='type', data=df, ax=ax_box, showfliers = False, dodge=False)
        # sns.kdeplot(data=predict, label="Predict", ax=ax_hist)
        # sns.kdeplot(data=test_data, label="Ground Truth", ax=ax_hist)
        # ax_box.set(xlabel='', ylabel='')
        # ax_hist.set(xlabel='Test set ratings')
        # plt.legend(labels=["Predict","Ground Truth"])
        # plt.legend()
        # plt.show()

    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label="Train")
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    path = join(get_plot_dir(), dataname+ f'_chunksize_{chunks}')
    plt.savefig(path)
    plt.show()
    print(RMSEs)
    # ml-1m: precision_acc,recall_acc:(0.055772734802054344, 0.025622311369641552)
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataname", choices=('ml-100k', 'ml-1m', 'jester', 'ml-10m'), default='ml-100k')
    parser.add_argument("--num_bt", type = int, required=False, default=100)
    parser.add_argument("--size_bt", type = int, required=False, default=1000)
    parser.add_argument("--percent", type=float, required=False, default = 0.5)
    parser.add_argument("--chunksize", type=float, required=False, default = 10)
    args = parser.parse_args()

    # np.random.seed(0)
    # norm_dist, ratings = generate_norm_dist(args.dataname)
    # ax = sns.kdeplot(norm_dist, label="Original")
    # plt.xlabel('ratings')
    
    
    ratings = load_rating_data(args.dataname)
    # PMF_visualizations(rating_norms, dataname, args.percent)
    # PMF_visualizations(ratings, dataname, args.percent)

    if args.dataname == "ml-1m":
        _lambda_U = _lambda_P = 0.045
    else:
        _lambda_U, _lambda_P = 10, 1000

    num_user = int(np.amax(ratings[:, 0])) + 1  # user总数
    num_item = int(np.amax(ratings[:, 1])) + 1  # movie总数

    for chunks in range(2, args.chunksize):
        ratings_chunks = data_chunks(ratings, chunks = 4)
        PMF_visual_chunks(ratings_chunks, num_user, num_item, args.percent, _lambda_U= _lambda_U, _lambda_P=_lambda_P,
                    num_batches=args.num_bt, batch_size=args.size_bt)

