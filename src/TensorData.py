import pandas as pd
import torch as t

class TensorData():
    def __init__(self, device, dataname, limit):
        self.device = device
        self.dataname = dataname
        self.limit = limit
        
    def tensor_3_dim_features(self):
        '''
        Desciption:
            Extracts the age, occupation, and gender features from the users. Here we label gender 'F' as
            0 and gender 'M' as 1. The index of occupations are from 0 to 20, and the index of ages is from 1 to 56.
        Input:
            limit: int 
                The limit number of data that would be process. Default is None, meaning having no limit
        Output:
            Array of ages, occupation, and genders of the users
        '''

        csv_users = pd.read_csv('data/users.csv', names = ["UserID", "Gender","Age","Occupation", "Zip-code"])
        df = pd.DataFrame(csv_users)
        if self.limit:
            df = df.head(self.limit)

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
            Gather csv files of Jester2 to retrievie the the numpy matrix of user-rating
        Output:
        A numpy matrix of user-rating
        '''
        if self.dataname == 'jester':
            ratings = pd.read_csv('data/jester.csv')
            # Eliminate the first column
            ratings = ratings.iloc[:,1:]
            tensor = pd.DataFrame(ratings).to_numpy()    
            tensor = t.tensor(tensor, dtype = t.float).to(self.device)

            # Change 99 into 0 and rescale the matrix by the fill value 
            observed_matrix = (tensor != 99)*1
            fill_value = t.abs(t.min(tensor)) + 1
            tensor = tensor + t.full(tensor.shape, fill_value = fill_value)
            tensor = t.mul(tensor, observed_matrix).to(self.device)

            return tensor

        elif self.dataname == 'ml-1m':
            ratings = pd.read_csv('data/ratings.csv', names = ["UserID", "MovieID","Rating","Timestamp"])
            df = pd.DataFrame(ratings)    
            sort_rating = df.sort_values(by = ['UserID', 'MovieID'], ignore_index = True)
            sort_rating_fill_0 = sort_rating.fillna(0)
            tensor = sort_rating_fill_0.pivot(index = 'UserID', columns = 'MovieID', values = 'Rating').fillna(0)
            tensor = t.tensor(tensor.to_numpy(), dtype = t.float).to(self.device)
            
            return tensor