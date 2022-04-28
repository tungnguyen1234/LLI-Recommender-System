#!/usr/bin/env python
'''
File descriptions
'''

__author__      = 'Tung Nguyen, Sang Truong'
__copyright__   = 'Copyright 2022, University of Missouri, Stanford University'

from argparse import Namespace, ArgumentParser
from matrix_movieLens import matrix_movieLens
from tensor_movieLens import tensor_movieLens

parser = ArgumentParser()

# general arguments
parser.add_argument("type", choices = ("matrix", "tensor"))
parser.add_argument("--percent", type=float, required=False, default = 0.2)
parser.add_argument("--eps", type=float, required=False, default = 1e-10)

# Configure for tensor
parser.add_argument("--limit", type=int, required=False, default = None)
parser.add_argument("--age", choices=('True', 'False'), default='False')
parser.add_argument("--occup", choices=('True', 'False'), default='False')
parser.add_argument("--gender", choices=('True', 'False'), default='False')

# JSON-like format
args = parser.parse_args()

if args.type == 'matrix':
    matrix_movieLens(args.percent, args.eps)

if args.type == 'tensor':
    args.features= set()
    if args.age == 'True':
        args.features.add("age")
    if args.occup == 'True':
        args.features.add("occup")
    if args.gender == 'True':
        args.features.add("gender")

    tensor_movieLens(args.features, args.percent, args.limit, args.eps)


