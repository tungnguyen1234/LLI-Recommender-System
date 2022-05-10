import pandas as pd
import torch as t

class TensorData():
    def __init__(self, device, dataname, limit):
        self.device = device
        self.dataname = dataname
        self.limit = limit
        
    def extract_features(self):
        if self.dataname == 'ml-1m':
            return self.tensor_features()

    def tensor_features(self):
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