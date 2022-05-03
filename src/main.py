#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace, ArgumentParser
from matrix_movieLens import matrix_movieLens
from matrix_Jester2 import matrix_Jester2
from tensor_movieLens import tensor_movieLens
import torch 

parser = ArgumentParser()
other_methods = ('svd', 'slopeone', 'mormpred', 'nmf', 'knn', 'knnmean', 'knnzscore', 'knnbaseline')

# general arguments
parser.add_argument("type", choices = ("matrix", "tensor"))
parser.add_argument("dataname", choices=('ml-1m', 'jester'), default='ml-1m')
parser.add_argument("--method", choices=other_methods, default='svd')
parser.add_argument("--percent", type=float, required=False, default = 0.2)
parser.add_argument("--eps", type=float, required=False, default = 1e-10)

# Configure for tensor
parser.add_argument("--limit", type=int, required=False, default = None)
parser.add_argument("--age", choices=('True', 'False'), default='False')
parser.add_argument("--occup", choices=('True', 'False'), default='False')
parser.add_argument("--gender", choices=('True', 'False'), default='False')
parser.add_argument("--gpuid", type=int)

# JSON-like format
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

if args.type == 'matrix':
    if args.dataname == 'ml-1m':
        matrix_movieLens(device, args.percent, args.eps)
    elif args.dataname == 'jester':
        matrix_Jester2(device, args.percent, args.eps)

if args.type == 'tensor' and args.dataname != 'jester':
    args.features= set()
    if args.age == 'True':
        args.features.add("age")
    if args.occup == 'True':
        args.features.add("occup")
    if args.gender == 'True':
        args.features.add("gender")

    tensor_movieLens(device, args.features, args.percent, args.limit, args.eps)
