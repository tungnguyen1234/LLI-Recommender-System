# Learning Latent Invariance (LLI)

Getting started
---------------
The LLI method is based on latent variable model. The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors.

Getting started
---------------
The main packages for LLI are ```PyTorch```, ```Numpy```, and ```Pandas```. 

We also use the [SurPRISE](http://surpriselib.com) python package to run other benchmark recommendation methods. The installation instruction is in the Installation section of the attached [link](https://github.com/NicolasHug/Surprise/tree/fa7455880192383f01475162b4cbd310d91d29ca)


Running matrix LLI
------------------

To run the matrix latent algorithm with train-test-split percentage and convergence condition:
```python src/main.py matrix --percent 0.2 --eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $10^{-5}$:
 ```python src/main.py matrix```

Evaluation of other matrix methods
----------------------------------


Running tensor LLI
------------------

For the tensor, there are 3 feature categories from the user: age, occupation, and gender. The 
command to run the tensor algorithm with train-test-split percentage, the number of limited data points, 
and all the possible features in the third dimension:
```python src/main.py tensor --percent 0.2 --limit 800```

or a simpler command is:
```python src/main.py tensor```

To run all data points with only the ```age``` feature:
```python src/main.py tensor --age True```

To add the feature ```occupation```:
```python src/main.py tensor --age True --occup True```

To run the feature ```gender```:
```python src/main.py tensor --gender True```

Evaluation of other tensor methods
----------------------------------

License
-------


Contributors
------------




