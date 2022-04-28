# Learning Latent Invariance (LLI)


For now, we access the file path ".../tensor_movieLens" to run the learning latent invariance for both matrix and tensor. 

The command to run the matrix latent algorithm with train-test-split percentage and convergence condition is is:

```python main.c matrix --percent 0.2 -eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $10^{-5}$:
 
 ```python main.c matrix```

For the tensor, there are 3 feature categories from the user: age, occupation, and gender. The 
command to run the tensor algorithm with train-test-split percentage, the number of limited data points, 
and all the possible features in the third dimension:

```python main.c tensor --percent 0.2 --limit 800```

or a simpler command is:

```python main.c tensor```

To run all data points with only the ```age``` feature, the command is:

```python main.c tensor --age True```

To add the feature ```occupation```, the command is:

```python main.c tensor --age True --occup True```
