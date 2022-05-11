#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

import torch as t, gc
from tqdm import tqdm
from TensorData import TensorData
from TensorObject import TensorObject

class TrainTest(TensorObject):
    def __init__(self, device, dim, feature, dataname, percent, limit):
        super().__init__(device, dim, dataname, percent, limit)
        '''
        Desciption:
            This class performs operations to create the tensor and splits the tensor for 
            training and testing.
        Input:
            .....
        '''

        data = TensorData(self.device, self.dataname, self.limit)
        self.feature = feature
        self.tensor_2_dim = data.tensor_2_dims()

        # for 3 dimension
        if self.dataname == 'ml-1m':
            self.ages, self.occupations, self.genders = data.tensor_3_dim_features()
        # Get the nonzero for faster process

        self.first_dim, self.second_dim = self.tensor_2_dim.shape
        self.user_dim = self.first_dim
        self.third_dim = 0


    def train_test(self):
        '''
        Desciption:
            Split the tensor into the training tensor and the index vectors for testing.
        Output:
            Returns the training tensor and the index vectors for testing.
        '''
        
        tensor_rating = None
        
        if self.dim == 2:
            tensor_rating = self.tensor_2_dim
        elif self.dim == 3:
            tensor_rating = self.get_tensor()
        
        sizes = tensor_rating.size()
        tensor_rating = t.flatten(tensor_rating)
        mask = (tensor_rating !=0)*1
        nonzero_mask = t.nonzero(mask)

        N = len(nonzero_mask)
        N_test = int(self.percent*N)
        
        # test indices
        idx_test = t.randperm(N)[:N_test]
        mask_test = t.zeros(mask.size()).to(self.device)
        mask_test[idx_test] = 1


        del idx_test, nonzero_mask, N, N_test
        tensor_train = t.reshape((mask - mask_test)*tensor_rating, sizes)
        mask_test = t.reshape(mask_test, sizes)


        return self.tensor_2_dim, tensor_train, mask_test



    def get_tensor(self):
        '''
        Desciption:
            Get the tensor depending the configured feature.
        Output:
            Returns the corresponding tensor based on the configured feature.
        '''
        if len(self.feature) == 1:
            if 'age' in self.feature:
                return self.tensor_age()
            if 'occup' in self.feature:
                return self.tensor_occup()
            if 'gender' in self.feature:
                return self.tensor_gender()
        elif len(self.feature) == 2:
            if self.feature == {'age', 'occup'}:
                return self.tensor_age_occup()
            if self.feature == {'age', 'gender'}:
                return self.tensor_age_gender()
            if self.feature == {'gender', 'occup'}:
                return self.tensor_gender_occup()
        else:
            return self.tensor_all()


    def tensor_age(self):
        '''
        Desciption:
            Extracts the tensor having ages as the third dimension. We construct the tensor
            by projecting the matrix rating of (user, product) pair into the respective tuple
            (user, product, age) in the tensor
        Output:
            Returns the tensor having the rating by (user, product, age) tuple
        '''

        # For Age: from 0 to 56 -> group 1 to 6. 
        third_dim = max(self.ages) + 1
        tensor_rating = t.zeros(self.first_dim, self.second_dim, third_dim).to(self.device)
        for user in tqdm(range(self.user_dim)):
            if user < len(self.ages):
                age = self.ages[user]
                tensor_rating[user, :, age] = self.tensor_2_dim[user, :]      
        return tensor_rating


    def tensor_occup(self):
        '''
        Desciption:
            Extracts the tensor having occupations as the third dimension. We construct the tensor
            by projecting the matrix rating of (user, product) pair into the respective tuple
            (user, product, occupation) in the tensor
        Output:
            Returns the tensor having the rating by (user, product, occupation) tuple
        '''


        # Get the dimensions 
        third_dim = max(self.occupations) + 1
        tensor_rating = t.zeros(self.first_dim, self.second_dim, third_dim).to(self.device)
        for user in tqdm(range(self.user_dim)):
            if user < len(self.occupations):
                occup = self.occupations[user]         
                tensor_rating[user, :, occup] = self.tensor_2_dim[user, :]
        return tensor_rating


    def tensor_gender(self):
        '''
        Desciption:
            Extracts the tensor having genders as the third dimension. We construct the tensor
            by projecting the matrix rating of (user, product) pair into the respective tuple
            (user, product, gender) in the tensor
        Output:
            Returns the tensor having the rating by (user, product, gender) tuple
        '''

        # Get the dimensions 
        third_dim = max(self.genders) + 1
        
        tensor_rating = t.zeros((self.first_dim, self.second_dim, third_dim)).to(self.device)
        for user in tqdm(range(self.user_dim)):
            if user < len(self.genders):
                gender = self.genders[user]         
                tensor_rating[user, :, gender] = self.tensor_2_dim[user, :]
        return tensor_rating

    def tensor_age_occup(self):
        '''
        Desciption:
            Extracts the tensor having ages and occupation as the third dimension. We construct the tensor
            by projecting the matrix rating of (user, product) pair into the respective tuples
            (user, product, age) and (user, product, occupation) in the tensor.
        Output:
            Returns the tensor having the rating by (user, product, age) 
                                    and (user, product, occupation) tuples
        '''

        # First group by occupation then age.
        tensor_age = self.tensor_age()
        tensor_occup = self.tensor_occup()

        tensor_rating = t.cat((tensor_age, tensor_occup), dim = 2).to(self.device)
        del tensor_occup, tensor_age
        t.cuda.empty_cache()
        return tensor_rating



    def tensor_age_gender(self):
        '''
        Desciption:
            Extracts the tensor having genders and ages as the third dimension. We construct the tensor
            by projecting the matrix rating of (user, product) pair into the respective tuples
            (user, product, gender) and (user, product, age) in the tensor.
        Output:
            Returns the tensor having the rating by (user, product, gender) and (user, product, age) tuples
        '''


        tensor_age = self.tensor_age()
        tensor_gender = self.tensor_gender()

        tensor_rating = t.cat((tensor_age, tensor_gender), dim = 2).to(self.device)
        del tensor_gender, tensor_age
        return tensor_rating


    def tensor_gender_occup(self):
        '''
        Desciption:
            Extracts the tensor having genders and occupations as the third dimension. 
            We construct the tensor by projecting the matrix rating of (user, product) pair into 
            the respective tuples (user, product, gender) and (user, product, occupation) in the tensor
        Output:
            Returns the tensor having the rating by (user, product, gender) and (user, product, occupation) tuples
        '''


        tensor_gender = self.tensor_gender()
        tensor_occup = self.tensor_occup()

        tensor_rating = t.cat((tensor_gender, tensor_occup), dim = 2).to(self.device)
        del tensor_gender, tensor_occup
        return tensor_rating



    def tensor_all(self):
        '''
        Desciption:
            Extracts the tensor having ages, genders, and occupations as the third dimension. 
            We construct the tensor by projecting the matrix rating of (user, product) pair into 
            the respective tuples (user, product, gender), (user, product, age), 
            and (user, product, occupation) in the tensor
        Output:
            Returns a bag having the indices by (user, product, gender), (user, product, age)
            and (user, product, occupation) tuples
        '''

        tensor_age_occup = self.tensor_age_occup()   
        tensor_gender = self.tensor_gender()
        tensor_rating = t.cat((tensor_age_occup, tensor_gender), dim = 2).to(self.device)
        gc.collect()
        t.cuda.empty_cache()
        return tensor_rating

    


