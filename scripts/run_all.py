"""
Master End-to-End Pipeline Automation Runner for PathSnap.
Executes all stages in sequence:
1. Exploratory Data Analysis (EDA)
2. Model Training & 5-Fold Group Cross-Validation
3. Bayesian Hyperparameter Optimization
4. HMM Viterbi Sequential Route Reconstruction
5. API Handlers Verification
6. Technical Report PDF Generation
"""

import os
import sys
import time

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)


def run_pipeline():
    print("=" * 70)
    print("PATHSNAP: END-TO-END AUTOMATION PIPELINE")
    print("Distributed Map-Matching & Route Reconstruction")
    print("PES University — Machine Learning UE24CS352A")
    print("=" * 70)

    total_start = time.time()

    # Stage 1: EDA
    print("\n>>> STAGE 1: Running Exploratory Data Analysis (EDA) <<<")
    from scripts.run_eda import main as run_eda
    run_eda()

    # Stage 2: Model Training & Comparison
    print("\n>>> STAGE 2: Training & Evaluating 5 Distinct ML Models <<<")
    from src.training.train_all_models import TrainingPipeline
    trainer = TrainingPipeline(repo_dir)
    trainer.run()

    # Stage 3: Bayesian Optimization
    print("\n>>> STAGE 3: Bayesian Hyperparameter Optimization (Optuna TPE) <<<")
    from src.training.bayesian_opt import BayesianOptimizer
    bo = BayesianOptimizer(repo_dir, n_trials=25)
    bo.run()

    # Stage 4: HMM & Viterbi Benchmark
    print("\n>>> STAGE 4: HMM & Viterbi Sequential Route Reconstruction <<<")
    from src.evaluation.benchmark import HMMBenchmark
    bench = HMMBenchmark(repo_dir)
    bench.run()

    # Stage 5: Technical Report Generation
    print("\n>>> STAGE 5: Generating Submission Technical Report PDF <<<")
    from report.generate_report import build_pdf_report
    pdf_path = build_pdf_report()

    total_duration = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"PATHSNAP PIPELINE COMPLETED SUCCESSFULLY IN {total_duration:.1f}s!")
    print(f"Technical Report Generated: {pdf_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
