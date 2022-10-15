# Learning Latent Invariance (LLI) for the Recommender System
The LLI algorithm extracts the latent variable vectors from using a linear optimization framework for tensor and retrieve recommendation from caculations of those latent variable vectors.

# Running 2D LLI
For train-test-split percentage $20%$ and convergence threshold $1e-10$ on MovieLens1M data:

```bash
python src/main.py 2 ml-1m --percent 0.2 --eps 1e-10 --gpuid 0
```

# Running 3D LLI
For the MovieLens1M dataset using 3 features with percentage $0.2$, $800$ data points, $10$ steps of evaluations, and GPU 0 is.

```bash
python src/main.py 3 ml-1m --percent 0.2 --limit 800 --num_feature 3 --steps 10 --gpuid 0
```
