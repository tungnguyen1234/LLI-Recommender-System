# -*- coding: utf-8 -*-
import numpy as np


class PMF(object):
    def __init__(self, num_feat=10, gamma=1, _lambda_U=0.1, _lambda_P=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.gamma = gamma  # learning rate,
        self._lambda_U = _lambda_U  # L2 regularization for User,
        self._lambda_P = _lambda_P  # L2 regularization for User,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)

        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []
        self.tests = []


    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#
    def fit(self, train_vec, test_vec):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  

        pairs_train = train_vec.shape[0]  # traindata 
        pairs_test = test_vec.shape[0]  # testdata

        # 1-p-i, 2-m-c
        num_user = int(max(np.amax(train_vec[:, 0]) , np.amax(test_vec[:, 0]))) + 1  # user总数
        num_item = int(max(np.amax(train_vec[:, 1]), np.amax(test_vec[:, 1]))) + 1  # movie总数

        incremental = False  
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0]) 
            np.random.shuffle(shuffled_order)  

            # Batch update
            for batch in range(self.num_batches):  
                # print("epoch %d batch %d" % (self.epoch, batch+1))

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda_U * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda_P * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.gamma * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.gamma * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda_U * np.linalg.norm(self.w_User) ** 2 \
                          + 0.5 * self._lambda_P *np.linalg.norm(self.w_Item) ** 2

                    self.rmse_train.append(np.sqrt(obj / pairs_train))

                # Compute validation error
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(test_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - test_vec[:, 2] + self.mean_inv
                    self.rmse_test.append(np.linalg.norm(rawErr) / np.sqrt(pairs_test))

                    # Print info
                    if batch == self.num_batches - 1:
                        print('Training RMSE: %f, Test RMSE %f' % (self.rmse_train[-1], self.rmse_test[-1]))


    def fit_train(self, train_vec, **kwargs):
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:, 2])  
        num_user, num_item = kwargs['num_user'], kwargs['num_item']
        pairs_train = train_vec.shape[0]  # traindata 

        incremental = False  
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(num_item, self.num_feat)  # numpy.random.randn 电影 M x D 
            self.w_User = 0.1 * np.random.randn(num_user, self.num_feat)  # numpy.random.randn 用户 N x D 

            self.w_Item_inc = np.zeros((num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0]) 
            np.random.shuffle(shuffled_order)  

            # Batch update
            for batch in range(self.num_batches):  
                # print("epoch %d batch %d" % (self.epoch, batch+1))

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv
                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda_U * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda_P * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((num_item, self.num_feat))
                dw_User = np.zeros((num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.gamma * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.gamma * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - train_vec[:, 2] + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda_U * np.linalg.norm(self.w_User) ** 2 \
                          + 0.5 * self._lambda_P *np.linalg.norm(self.w_Item) ** 2

                    self.rmse_train.append(np.sqrt(obj / pairs_train))
                    # print('Training RMSE: %f' % (self.rmse_train[-1]))

                    # Compute validation error
                    self.tests.append([np.copy(self.w_User), np.copy(self.w_Item)])
                        
    def get_RMSE_test(self, test_vec, pairs_test):
        for w_User, w_Item in self.tests:
            pred_out = np.sum(np.multiply(
                            w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                            w_Item[np.array(test_vec[:, 1], dtype='int32'), :]
                            ), axis=1)  # mean_inv subtracted
            rawErr = pred_out + self.mean_inv - test_vec[:, 2]
            # rmse = np.linalg.norm(rawErr) / np.sqrt(pairs_test)
            rmse = np.linalg.norm(rawErr, ord=1) / pairs_test
            self.rmse_test.append(rmse)

    def predict_ID(self, invID):
        re = np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot
        return re

    def predict_full(self): 
        return self.w_User @ self.w_Item.T + self.mean_inv

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, **parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.gamma = parameters.get("gamma", 1)
            self._lambda_U = parameters.get("_lambda_U", 0.1)
            self._lambda_P = parameters.get("_lambda_P", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def get_params(self, deep=True):
        # Return a dictionary of the model's hyperparameters
        return {"num_feat": self.num_feat, "gamma": self.gamma, "_lambda_U": self._lambda_U, 
                "_lambda_P": self._lambda_P, "momentum": self.momentum, "maxepoch": self.maxepoch, 
                "num_batches": self.num_batches, "batch_size": self.batch_size}

    def test_ratings(self, test_vec):
        full_pred = self.predict_full()
        test_result = []
        for uid, mid, _ in test_vec:
            test_result.append(full_pred[int(uid), int(mid)])
        return test_result

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            inv = int(inv)
            if pred.get(inv, None) is None:
                predict = self.predict_ID(inv)
                pred[inv] = np.argsort(predict)[-k:]  # numpy.argsort

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                u = int(test_vec[i, 0])
                intersection_cnt[u] = intersection_cnt.get(u, 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))
        

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        result = [precision_acc / len(inv_lst), recall_acc / len(inv_lst)]
        del pred, intersection_cnt, precision_acc, recall_acc
        return result




class PMF_GS(object):
    def __init__(self, num_feat=10, gamma=1, 
        _lambda_U=0.1, _lambda_P=0.1, momentum=0.8, maxepoch=20, num_batches=10, batch_size=1000):
        self.num_feat = num_feat  # Number of latent features,
        self.gamma = gamma  # learning rate,
        self._lambda_U = _lambda_U  # L2 regularization for User,
        self._lambda_P = _lambda_P  # L2 regularization for User,
        self.momentum = momentum  # momentum of the gradient,
        self.maxepoch = maxepoch  # Number of epoch before stop,
        self.num_batches = num_batches  # Number of batches in each epoch (for SGD optimization),
        self.batch_size = batch_size  # Number of training samples used in each batches (for SGD optimization)
        self.w_User = None  # User feature vectors

        self.rmse_train = []
        self.rmse_test = []
        self.tests = []

    # ***Fit the model with train_tuple and evaluate RMSE on both train and test data.  ***********#
    # ***************** train_vec=TrainData, test_vec=TestData*************#

    def fit(self, train_vec, y_vec, **kwargs):
        # mean subtraction
        self.mean_inv = np.mean(y_vec)  
        pairs_train = train_vec.shape[0]  # traindata
        self.num_user, self.num_item = kwargs['num_user'], kwargs['num_item']

        incremental = False  
        if ((not incremental) or (self.w_Item is None)):
            # initialize
            self.epoch = 0
            self.w_Item = 0.1 * np.random.randn(self.num_item, self.num_feat)  # numpy.random.randn 电影 M x D 
            self.w_User = 0.1 * np.random.randn(self.num_user, self.num_feat)  # numpy.random.randn 用户 N x D 

            self.w_Item_inc = np.zeros((self.num_item, self.num_feat))  # 创建电影 M x D 0矩阵
            self.w_User_inc = np.zeros((self.num_user, self.num_feat))  # 创建用户 N x D 0矩阵

        while self.epoch < self.maxepoch:  
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0]) 
            np.random.shuffle(shuffled_order)  

            # Batch update
            for batch in range(self.num_batches):  
                # print("epoch %d batch %d" % (self.epoch, batch+1))

                test = np.arange(self.batch_size * batch, self.batch_size * (batch + 1))
                batch_idx = np.mod(test, shuffled_order.shape[0])  

                batch_UserID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_ItemID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_User[batch_UserID, :],
                                              self.w_Item[batch_ItemID, :]),
                                  axis=1)  # mean_inv subtracted # np.multiply

                rawErr = pred_out - y_vec[shuffled_order[batch_idx]] + self.mean_inv

                # Compute gradients
                Ix_User = 2 * np.multiply(rawErr[:, np.newaxis], self.w_Item[batch_ItemID, :]) \
                       + self._lambda_U * self.w_User[batch_UserID, :]
                Ix_Item = 2 * np.multiply(rawErr[:, np.newaxis], self.w_User[batch_UserID, :]) \
                       + self._lambda_P * (self.w_Item[batch_ItemID, :])  # np.newaxis :increase the dimension

                dw_Item = np.zeros((self.num_item, self.num_feat))
                dw_User = np.zeros((self.num_user, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_Item[batch_ItemID[i], :] += Ix_Item[i, :]
                    dw_User[batch_UserID[i], :] += Ix_User[i, :]

                # Update with momentum
                self.w_Item_inc = self.momentum * self.w_Item_inc + self.gamma * dw_Item / self.batch_size
                self.w_User_inc = self.momentum * self.w_User_inc + self.gamma * dw_User / self.batch_size

                self.w_Item = self.w_Item - self.w_Item_inc
                self.w_User = self.w_User - self.w_User_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    pred_out = np.sum(np.multiply(self.w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                                                  self.w_Item[np.array(train_vec[:, 1], dtype='int32'), :]),
                                      axis=1)  # mean_inv subtracted
                    rawErr = pred_out - y_vec + self.mean_inv
                    obj = np.linalg.norm(rawErr) ** 2 \
                          + 0.5 * self._lambda_U * np.linalg.norm(self.w_User) ** 2 \
                          + 0.5 * self._lambda_P *np.linalg.norm(self.w_Item) ** 2
                    # obj = np.linalg.norm(rawErr, ord=1) \
                    #       + 0.5 * self._lambda_U * np.linalg.norm(self.w_User) ** 2 \
                    #       + 0.5 * self._lambda_P *np.linalg.norm(self.w_Item) ** 2

                    self.rmse_train.append(np.sqrt(obj / pairs_train))
                    # print('Training RMSE: %f' % (self.rmse_train[-1]))

                    # Compute validation error
                    self.tests.append([np.copy(self.w_User), np.copy(self.w_Item)])

    def score(self, train_vec, y_train):
        pairs = train_vec.shape[0]  # testdata
        w_User, w_Item = self.tests[-1]
        pred_out = np.sum(np.multiply(
                        w_User[np.array(train_vec[:, 0], dtype='int32'), :],
                        w_Item[np.array(train_vec[:, 1], dtype='int32'), :]
                        ), axis=1)  # mean_inv subtracted
        rawErr = pred_out + self.mean_inv - y_train
        # rmse = np.linalg.norm(rawErr) / np.sqrt(pairs_test)
        rmse = np.linalg.norm(rawErr, ord=1) / pairs
        return rmse         

    def test_RMSE(self, test_vec, y_test):
        pairs_test = test_vec.shape[0]  # testdata
        for w_User, w_Item in self.tests:
            pred_out = np.sum(np.multiply(
                            w_User[np.array(test_vec[:, 0], dtype='int32'), :],
                            w_Item[np.array(test_vec[:, 1], dtype='int32'), :]
                            ), axis=1)  # mean_inv subtracted
            rawErr = pred_out + self.mean_inv - y_test
            # rmse = np.linalg.norm(rawErr) / np.sqrt(pairs_test)
            rmse = np.linalg.norm(rawErr, ord=1) / pairs_test
            self.rmse_test.append(rmse)
        return self.rmse_test[-1]

    def predict_ID(self, invID):
        re = np.dot(self.w_Item, self.w_User[int(invID), :]) + self.mean_inv  # numpy.dot
        return re

    def predict(self, X): 
        return self.w_User @ self.w_Item.T + self.mean_inv

    # ****************Set parameters by providing a parameter dictionary.  ***********#
    def set_params(self, **parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.gamma = parameters.get("gamma", 1)
            self._lambda_U = parameters.get("_lambda_U", 0.1)
            self._lambda_P = parameters.get("_lambda_P", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)
        return self

    def get_params(self, deep=True):
        # Return a dictionary of the model's hyperparameters
        return {"num_feat": self.num_feat, "gamma": self.gamma, "_lambda_U": self._lambda_U, 
                "_lambda_P": self._lambda_P, "momentum": self.momentum, "maxepoch": self.maxepoch, 
                "num_batches": self.num_batches, "batch_size": self.batch_size}

    def test_ratings(self, test_vec):
        full_pred = self.predict_full()
        test_result = []
        for uid, mid, _ in test_vec:
            test_result.append(full_pred[int(uid), int(mid)])
        return test_result

    def topK(self, test_vec, k=10):
        inv_lst = np.unique(test_vec[:, 0])
        pred = {}
        for inv in inv_lst:
            inv = int(inv)
            if pred.get(inv, None) is None:
                predict = self.predict_ID(inv)
                pred[inv] = np.argsort(predict)[-k:]  # numpy.argsort

        intersection_cnt = {}
        for i in range(test_vec.shape[0]):
            if test_vec[i, 1] in pred[test_vec[i, 0]]:
                u = int(test_vec[i, 0])
                intersection_cnt[u] = intersection_cnt.get(u, 0) + 1
        invPairs_cnt = np.bincount(np.array(test_vec[:, 0], dtype='int32'))
        

        precision_acc = 0.0
        recall_acc = 0.0
        for inv in inv_lst:
            precision_acc += intersection_cnt.get(inv, 0) / float(k)
            recall_acc += intersection_cnt.get(inv, 0) / float(invPairs_cnt[int(inv)])

        result = [precision_acc / len(inv_lst), recall_acc / len(inv_lst)]
        del pred, intersection_cnt, precision_acc, recall_acc
        return result


