"""
Bayesian Hyperparameter Optimization Runner Script.
"""
import os, sys
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
from src.training.bayesian_opt import BayesianOptimizer

if __name__ == "__main__":
    bo = BayesianOptimizer(repo_dir, n_trials=25)
    bo.run()
