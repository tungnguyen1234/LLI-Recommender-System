import pandas as pd
import torch as t
from os.path import join
from builtin_datasets import get_dataset_dir
from reader import ReaderFeatures
from dataset import Dataset

# https://keras.io/examples/keras_recipes/memory_efficient_embeddings/

class TensorData():
    def __init__(self, device, dataname, limit):
        '''
        Input:
            limit: The limit number of data that would be process. 
                    Default is None, meaning having no limit
            device: GPU device
            dataname: the dataset name that is used
        '''
        self.device = device
        self.dataname = dataname
        self.limit = limit
        
    def tensor_3_dim_features(self):
        '''
        Desciption:
            Extracts the age, occupation, and gender features for ml-1m dataset. Here we label gender 'F' as
            0 and gender 'M' as 1. The index of occupations are from 0 to 20, and the index of ages is from 1 to 56.
        Output:
            Torch array of ages, occupation, and gender.
        '''
        file_path = join(get_dataset_dir() + '/ml-1m/ml-1m/users.dat')
        reader = ReaderFeatures(line_format='id gender age occupation zip', sep='::')

        data = Dataset.load_features_from_file(file_path, reader=reader)
        df = pd.DataFrame(data.raw_features, columns = ["UserID", "Gender","Age","Occupation", "Zip-code"])

        # Get age and profile info
        ages = t.tensor(df['Age'].to_numpy() // 10, dtype = t.int)

        # Job
        occupations = t.tensor(df['Occupation'].to_numpy())

        # Gender
        genders = []
        for gender in list(df['Gender']):
            if gender == 'F':
                genders.append(1)
            elif gender == 'M':
                genders.append(0)

        genders = t.tensor(genders)
        return ages, occupations, genders

    def tensor_2_dims(self):
        '''
        Desciption:
            Gather csv files of all datasets from .dat files
            to retrievie the torch matrix of user-product-ratings
        Output:
            A torch matrix of user-rating
        '''
        
        data = Dataset.load_builtin(self.dataname)
        df = pd.DataFrame(data.raw_ratings, columns = ["UserID", "ProductID","Rating","Timestamp"])
        if self.limit:
            df = df.sample(n = self.limit)
        sort_rating = df.sort_values(by = ['UserID', 'ProductID'], ignore_index = True)
        tensor = sort_rating.pivot(index = 'UserID', columns = 'ProductID', values = 'Rating').fillna(float('inf'))
        tensor = t.tensor(tensor.to_numpy(), dtype = t.float).to(self.device)                                                                                                                                                                                                                                                                                                                                                                            

        if self.dataname == 'jester':
            fill_value = t.abs(t.min(tensor)) + 1
            tensor = tensor + t.full(tensor.shape, fill_value = fill_value)
            
        tensor[tensor == float('inf')] = 0
        # tensor = tensor * 5
        return tensor
