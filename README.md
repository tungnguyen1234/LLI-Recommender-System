# Learning Latent Invariance (LLI)

## Usage
To run the matrix latent algorithm with train-test-split percentage and convergence condition:
```python src/main.py matrix --percent 0.2 --eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $10^{-5}$:
 ```python src/main.py matrix```

For the tensor, there are 3 feature categories from the user: age, occupation, and gender. The 
command to run the tensor algorithm with train-test-split percentage, the number of limited data points, 
and all the possible features in the third dimension:
```python src/main.py tensor --percent 0.2 --limit 800```

or a simpler command is:
```python src/main.py tensor```


Overall, we could reproduce experiements in movielens by the following operations:
Adding one feature:

```python src/main.py tensor --age True```

```python src/main.py tensor --occup True```

```python src/main.py tensor --gender True```

Adding two features:

```python src/main.py tensor --age True --gender True```

```python src/main.py tensor --age True --occup True```

```python src/main.py tensor --gender True --occup True```


Adding three features:

```python src/main.py tensor```

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
