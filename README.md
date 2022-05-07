# Learning Latent Invariance (LLI)

# Getting started

The LLI method is based on latent variable model. The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors.

# Packages

The main packages for LLI are ```PyTorch```, ```Numpy```, and ```Pandas```. 

We also use the [SurPRISE](http://surpriselib.com) python package to run other benchmark recommendation methods. The installation instruction is in the Installation section of the attached [link](https://github.com/NicolasHug/Surprise/tree/fa7455880192383f01475162b4cbd310d91d29ca)

# Parser arguments

For the tensor, we have the following parser argument:

* ```type```: matrix or tensor for evaluation.
* ```dataname```: the dataset to use. Here we only include ```ml-1m``` (MovieLens1M) and ```jester``` (Jester2).
* ```--method```: indicating the method to evaluate the dataset. Default is ```LLI```.
* ```--percent```: the ratio to divide between training and testing set. Default is ```0.2```.
* ```--eps```: convergence threshold for the LLI algorithm. Default is ```1e-5```
* ```--steps```: the number of evaluations on the MAE and RMSE for the LLI algorithm. Default is ```10```.
* ```--limit```: the number of data to limit for debugging purpose. Default is ```None```, meaning full dataset.
* ```--num_feature```: the number of features to add in the third dimension of the tensor. Default is ```3```.
* ```--gpuid```: the index number of the GPU. 


# Running matrix LLI

To run the matrix LLI algorithm with train-test-split percentage $20%$ and convergence threshold $1e-10$ on MovieLens1M data:

```python src/main.py matrix ml-1m --method LLI --percent 0.2 --eps 1e-10```

The command is also self-configured with a 20% percentage and convergence condition of $1e-10$:

```python src/main.py matrix ml-1m```

The same command for Jester2 dataset is 

```python src/main.py matrix jester```

# Evaluation of other matrix methods


# Running tensor LLI

* This command runs the LLI method on the MovieLens1M dataset with percentage $0.2$, $800$ data points, $10$ steps of evaluations, and GPU 0 is.

```python src/main.py tensor ml-1m --method LLI --percent 0.2 --limit 800 --steps 10 --gpuid 0```

* A simpler command is:

```python src/main.py tensor ml-1m --gpuid 0.```

* A similar command without using GPU id is:

```python src/main.py tensor ml-1m.```

* Also, Jester-2 dataset can only run in matrix, but not in tensor. 

# Commands to evaluate the tensor algorithm based on the features added.

Adding one feature:

```python src/main.py tensor ml-1m --num_feature 1```

Adding two features:

```python src/main.py tensor ml-1m --num_feature 2```

Adding three features is either

```python src/main.py tensor ml-1m --num_feature 3```

or 

```python src/main.py tensor ml-1m```

# Evaluation of other tensor methods


# License


# Citation
```
@misc{
    nguyen2022LLI,
    title={Learning Latent Invariant with Tensor Decomposition for Recommender Systems},
    author={Tung Nguyen, Sang Truong, and Jeffrey Uhlmann},
    publisher={arXiv},
    year={2022},
}
```
