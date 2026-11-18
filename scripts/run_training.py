"""
Model Training Runner Script.
"""
import os, sys
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
from src.training.train_all_models import TrainingPipeline

if __name__ == "__main__":
    p = TrainingPipeline(repo_dir)
    p.run()
