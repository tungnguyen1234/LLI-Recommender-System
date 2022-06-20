#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import ArgumentParser
from Tensor import Tensor
import torch as t

parser = ArgumentParser()

# general arguments
parser.add_argument("dim", type = int, choices = (2, 3))
parser.add_argument("dataname", choices=('ml-1m', 'jester', 'ml-10m'), default='ml-1m')
parser.add_argument("--percent", type=float, required=False, default = 0.2)
parser.add_argument("--eps", type=float, required=False, default = 1e-10)
parser.add_argument("--steps", type = int, required=False, default=10)

# Configure for tensor
parser.add_argument("--limit", type=int, required=False, default = None)
parser.add_argument("--num_feature", type=int, required=False, default = 3)
parser.add_argument("--gpuid", type=int, required=False, default = None)

# JSON-like format
args = parser.parse_args()
device = t.device(f"cuda:{args.gpuid}" if t.cuda.is_available() else "cpu")
tensor = Tensor(device, args.dim, args.dataname, args.num_feature, args.percent, \
    args.eps, args.steps, args.limit)

tensor.performance_overal_LLI()    
