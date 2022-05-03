# Learning Latent Invariance (LLI)

Getting started
---------------
The LLI method is based on latent variable model. The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors.

Packages
--------
The main packages for LLI are ```PyTorch```, ```Numpy```, and ```Pandas```. 

We also use the [SurPRISE](http://surpriselib.com) python package to run other benchmark recommendation methods. The installation instruction is in the Installation section of the attached [link](https://github.com/NicolasHug/Surprise/tree/fa7455880192383f01475162b4cbd310d91d29ca)

Running matrix LLI
------------------

To run the matrix LLI algorithm with train-test-split percentage and convergence condition on MovieLens1M data:

```python src/main.py matrix ml-1m --method LLI --percent 0.2 --eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $10^{-5}$:

```python src/main.py matrix ml-1m```

The same command for Jester2 dataset is 

```python src/main.py matrix jester```

Evaluation of other matrix methods
----------------------------------


Running tensor LLI
------------------

For the tensor, there are 3 feature categories from the user: age, occupation, and gender. The 
command to run the tensor LLI algorithm with train-test-split percentage, the number of limited data points, 
and all the possible features in the third dimension of MovieLens1M data:

```python src/main.py tensor ml-1m --method LLI --percent 0.2 --limit 800```

or a simpler command is:

```python src/main.py tensor ml-1m```

The same command for Jester2 dataset is 

```python src/main.py tensor jester```


Overall, we could reproduce experiements in tensor movielens by the following operations with GPU ID on only ```ml-1m``` dataset. Jester-2 dataset can only run in matrix, but not in tensor:

Adding one feature:

```python src/main.py tensor ml-1m --age True```

```python src/main.py tensor ml-1m --occup True```

```python src/main.py tensor ml-1m --gender True```

Adding two features:

```python src/main.py tensor ml-1m --age True --gender True```

```python src/main.py tensor ml-1m --age True --occup True```

```python src/main.py tensor ml-1m --gender True --occup True```


Adding three features:

```python src/main.py tensor ml-1m```

Evaluation of other tensor methods
----------------------------------

License
-------


Citation
------------
```
@misc{
    nguyen2022LLI,
    title={Learning Latent Invariant with Tensor Decomposition for Recommender Systems},
    author={Tung Nguyen, Sang Truong, and Jeffrey Uhlmann},
    publisher={arXiv},
    year={2022},
}
```
