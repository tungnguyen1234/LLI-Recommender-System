# Learning Latent Invariance (LLI)

# Getting started

* The LLI method is based on latent variable model. The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors.

# Packages

* The main packages for LLI are ```PyTorch```, ```Numpy```, and ```Pandas```. We also use the [SurPRISE](http://surpriselib.com) python package to run other benchmark recommendation methods. The installation instruction is in the Installation section of the attached [link](https://github.com/NicolasHug/Surprise/tree/fa7455880192383f01475162b4cbd310d91d29ca)

# Running matrix LLI

* To run the 2D LLI algorithm with train-test-split percentage $20%$ and convergence threshold $1e-10$ on MovieLens1M data:

```python src/main.py 2 ml-1m --method LLI --percent 0.2 --eps 1e-10 --gpuid 1```

# Running tensor LLI

* This command runs the LLI method on the MovieLens1M dataset with percentage $0.2$, $800$ data points, $10$ steps of evaluations, and GPU 0 is.

```python src/main.py 3 ml-1m --method LLI --percent 0.2 --limit 800 --steps 10 --gpuid 0```

or a shorter/simpler command is:

```python src/main.py 3 ml-1m```


# Evaluate the tensor algorithm (need to run!!)

* For example, to run 3D LLI and there is 2 features:
```python src/main.py 3 ml-1m --num_feature 1 --gpuid 0```

```python src/main.py 3 ml-1m --num_feature 2 --gpuid 0```

```python src/main.py 3 ml-1m --num_feature 3 --gpuid 0```


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
