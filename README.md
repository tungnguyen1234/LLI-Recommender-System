# Learning Latent Invariance (LLI) for Recommender System

# Getting started

* The LLI method is based on latent variable model. The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors. The current research paper is in progress.

# Packages

* The main packages for LLI are ```PyTorch```, ```Numpy```, and ```Pandas```. We also test other benchmarks using the [SurPRISE](http://surpriselib.com) package.
# Running matrix LLI

* To run the 2D LLI algorithm with train-test-split percentage $20%$ and convergence threshold $1e-10$ on MovieLens1M data:

```bash
python src/main.py 2 ml-1m --percent 0.2 --eps 1e-10 --gpuid 0
```

* A shorter/simpler command is:

```bash
python src/main.py 2 ml-1m
```

# Running tensor LLI

* This command runs the LLI method on the MovieLens1M dataset with percentage $0.2$, $800$ data points, $10$ steps of evaluations, and GPU 0 is.

```bash
python src/main.py 3 ml-1m --percent 0.2 --limit 800 --steps 10 --gpuid 0
```

* A shorter/simpler command is:

```bash
python src/main.py 3 ml-1m
```


# To run the command with GPU and some number of features, we could do the following:

```bash 
python src/main.py 3 ml-1m --num_feature 2 --gpuid 0
```
