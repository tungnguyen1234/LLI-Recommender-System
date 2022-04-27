#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace, 
import pandas as pd
import numpy as np

parser = ArgumentParser()

# general arguments
parser.add_argument("--age", choices=("True", "False"), default="True")

args = parser.parse_args()
args.amortized_vi = True if args.amortized_vi == "True" else False



matrix_rating = matrix_rating()
ages, occupations = extract_3D_dataset(900)

# matrix_rating = np.array([[1, 1, 0], [0, 0, 2], [3, 3, 4]])
# ages = np.array([1, 20, 30])
# occupations = np.array([0, 4, 5, 6])
# print(len(ages))

tensor_rating = tensor_age(matrix_rating, ages)
# print(tensor_rating)
# tensor_rating = tensor_occupation(matrix_rating, occupations)
# print(tensor_rating)
# tensor_rating = tensor_age_occup(matrix_rating, ages, occupations)
# print(tensor_rating)


MAE, MSE, errors = tensor_traintest_score(tensor_rating, percent= 0.2)
print("MAE is", round(MAE, 2))
print("MSE is", round(MSE, 2))
print("Error intervals has", errors)
