"""
HMM Viterbi Evaluation Benchmark Runner Script.
"""
import os, sys
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path: sys.path.insert(0, repo_dir)
from src.evaluation.benchmark import HMMBenchmark

if __name__ == "__main__":
    bench = HMMBenchmark(repo_dir)
    bench.run()
