#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import ArgumentParser
from Tensor import Tensor
from OtherMethods import other_methods
import torch 

parser = ArgumentParser()
other_methods = ('LLI', 'svd', 'slope_one', 'norm_pred', 'nmf', 'knn_basic', 'knn_with_means', 'knn_with_z_score', 'knn_baseline')

# general arguments
parser.add_argument("dim", type = int, choices = (2, 3))
parser.add_argument("dataname", choices=('ml-1m', 'jester'), default='ml-1m')
parser.add_argument("--method", choices=other_methods, default='LLI')
parser.add_argument("--percent", type=float, required=False, default = 0.2)
parser.add_argument("--eps", type=float, required=False, default = 1e-10)
parser.add_argument("--steps", type = int, required=False, default=10)

# Configure for tensor
parser.add_argument("--limit", type=int, required=False, default = None)
parser.add_argument("--num_feature", type=int, required=False, default = 0)
parser.add_argument("--gpuid", type=int)

# JSON-like format
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu")

if args.dim == 3 and args.num_feature == 0:
    args.num_feature = 3

tensor = Tensor(device, args.dim, args.dataname, args.method, args.num_feature, args.percent, args.eps, args.steps, args.limit)

if args.method == 'LLI':
    tensor.performance_overal_LLI()
elif args.dim == 2:
    other_methods(args.percent, args.dataname, args.method)        
