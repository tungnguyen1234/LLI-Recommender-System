# Learning Latent Invariance (LLI)


For now, we access file path ".../tensor_movieLens" to run the matrix and tensor latent scaling. 

The command to run the matrix latent algorithm with train-test-split percentage is:
```python main.c matrix --percent 0.2```

For the tensor, we add only 3 feature labels for the user: age, occupation, and gender.
The command to run the tensor case with train-test-split percentage, and number of limited data points, and all features
in the third dimension is:

```python main.c tensor --percent 0.2 --limit 800```

If we do not limit the number of data points and run only the ```age``` feature, the command is
```python main.c tensor --percent 0.2 --age True```

If we also add feature ```occupation``` to the label, the command is:
```python main.c tensor --percent 0.2 --age True --occup True```