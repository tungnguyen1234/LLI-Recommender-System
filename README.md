# Learning Latent Invariance (LLI)

## Usage
To run the matrix latent algorithm with train-test-split percentage and convergence condition on MovieLens1M data:
```python src/main.py matrix dataname ml-1m --percent 0.2 --eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $10^{-5}$:
```python src/main.py matrix dataname ml-1m```

The same command for Jester2 dataset is 
```python src/main.py matrix dataname jester```

Evaluation of other matrix methods
----------------------------------


Running tensor LLI
------------------

For the tensor, there are 3 feature categories from the user: age, occupation, and gender. The 
command to run the tensor algorithm with train-test-split percentage, the number of limited data points, 
and all the possible features in the third dimension of MovieLens1M data:
```python src/main.py tensor ml-1m --percent 0.2 --limit 800```

or a simpler command is:
```python src/main.py tensor ml-1m```

The same command for Jester2 dataset is 
```python src/main.py tensor jester```


Overall, we could reproduce experiements in tensor movielens by the following operations with GPU ID, and the similar commands go for Jester-2 dataset as ```jester```:

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

## Citation
```
@misc{
    nguyen2022LLI,
    title={Learning Latent Invariant with Tensor Decomposition for Recommender Systems},
    author={Tung Nguyen, Sang Truong, and Jeffrey Uhlmann},
    publisher={arXiv},
    year={2022},
}
```
