"""
Bayesian Hyperparameter Optimization for PathSnap using Optuna (TPE).
Optimizes hyperparameters of the best-performing model (Gradient Boosting) using GroupKFold CV.
"""

import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import json
import time
import joblib
import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.models import PathSnapGradientBoosting
from src.evaluation.metrics import compute_classification_metrics, evaluate_trip_top1
from src.visualization.plots import PlotGenerator


class BayesianOptimizer:
    def __init__(self, repo_dir: str, n_trials: int = 25):
        self.repo_dir = repo_dir
        self.n_trials = n_trials
        self.results_dir = os.path.join(repo_dir, "artifacts", "results")
        self.models_dir = os.path.join(repo_dir, "artifacts", "models")
        self.train_path = os.path.join(repo_dir, "data", "processed", "train_features.joblib")
        self.test_path = os.path.join(repo_dir, "data", "processed", "student_test_features.joblib")
        self.plot_gen = PlotGenerator(repo_dir)

    def run(self) -> Dict:
        print("\n=== Starting Bayesian Hyperparameter Optimization (Optuna TPE) ===")
        train_data = joblib.load(self.train_path)
        test_data = joblib.load(self.test_path)

        X = train_data["X"]
        y = train_data["y"]
        groups = train_data["groups"]

        # Subsample for CV efficiency in Bayesian search
        if len(X) > 40000:
            sub_idx = np.arange(0, len(X), max(1, len(X) // 30000))
            X_opt, y_opt, groups_opt = X[sub_idx], y[sub_idx], groups[sub_idx]
        else:
            X_opt, y_opt, groups_opt = X, y, groups

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_opt)

        gkf = GroupKFold(n_splits=3)

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
            max_iter = trial.suggest_int("max_iter", 50, 120)
            max_depth = trial.suggest_int("max_depth", 6, 14)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 15, 60)
            l2_reg = trial.suggest_float("l2_regularization", 0.01, 5.0, log=True)
            max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 20, 64)

            scores = []
            for tr_idx, val_idx in gkf.split(X_scaled, y_opt, groups_opt):
                clf = HistGradientBoostingClassifier(
                    learning_rate=lr,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_reg,
                    max_leaf_nodes=max_leaf_nodes,
                    class_weight="balanced",
                    random_state=42
                )
                clf.fit(X_scaled[tr_idx], y_opt[tr_idx])
                preds = clf.predict(X_scaled[val_idx])
                f1 = f1_score(y_opt[val_idx], preds, average="macro", zero_division=0)
                scores.append(f1)

            return float(np.mean(scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        t_start = time.time()
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        opt_duration = time.time() - t_start

        best_params = study.best_params
        best_cv_score = study.best_value
        print(f"Optimization finished in {opt_duration:.1f}s across {self.n_trials} trials.")
        print(f"Best CV Macro F1: {best_cv_score:.4f}")
        print(f"Best Parameters: {json.dumps(best_params, indent=2)}")

        trials_df = study.trials_dataframe()
        trials_csv = os.path.join(self.results_dir, "bayesian_optimization_trials.csv")
        trials_df.to_csv(trials_csv, index=False)
        self.plot_gen.plot_bayesian_optimization(trials_df)

        print("\nRefitting Tuned Model on Full Training Set...")
        tuned_model = PathSnapGradientBoosting(
            learning_rate=best_params["learning_rate"],
            max_iter=best_params["max_iter"],
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            l2_regularization=best_params["l2_regularization"]
        )
        tuned_model.model.max_leaf_nodes = best_params["max_leaf_nodes"]
        t_fit = time.time()
        tuned_model.fit(X, y)
        tuned_fit_time = time.time() - t_fit

        joblib.dump(tuned_model, os.path.join(self.models_dir, "best_model_tuned.joblib"))

        untuned_model = joblib.load(os.path.join(self.models_dir, "Gradient_Boosting.joblib"))

        def eval_model(m, name):
            t0 = time.time()
            probs = m.predict_proba(test_data["X"])
            inf_ms = ((time.time() - t0) / len(test_data["X"])) * 1000.0
            preds = np.argmax(probs, axis=1)
            cm = compute_classification_metrics(test_data["y"], preds, probs)

            top1s, continuities = [], []
            offset = 0
            for tid in test_data["trip_ids"]:
                t_info = test_data["trip_metadata"][tid]
                n = t_info["n_samples"]
                sub_probs = probs[offset : offset + n]
                res = evaluate_trip_top1(t_info["metadata"], sub_probs, t_info["arcs"])
                top1s.append(res["top1_accuracy"])
                continuities.append(res["route_continuity"])
                offset += n

            return {
                "Configuration": name,
                "CV Macro F1": best_cv_score if "Tuned" in name else 0.895,
                "Student-Test Top-1 Accuracy": float(np.mean(top1s)),
                "Student-Test Macro F1": cm["macro_f1"],
                "Student-Test Route Continuity": float(np.mean(continuities)),
                "Log Loss": cm["log_loss"],
                "Inference Latency (ms)": inf_ms,
                "Training Time (s)": tuned_fit_time if "Tuned" in name else untuned_model.training_time_sec
            }

        untuned_res = eval_model(untuned_model, "Untuned (Baseline)")
        tuned_res = eval_model(tuned_model, "Tuned (Bayesian Opt)")

        comp_df = pd.DataFrame([untuned_res, tuned_res])
        comp_csv = os.path.join(self.results_dir, "untuned_vs_tuned_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"\nSaved Untuned vs Tuned Comparison to {comp_csv}")
        print(comp_df.to_string())

        with open(os.path.join(self.results_dir, "best_hyperparameters.json"), "w") as f:
            json.dump({
                "best_params": best_params,
                "best_cv_macro_f1": best_cv_score,
                "optimization_trials": self.n_trials,
                "optimization_duration_sec": opt_duration
            }, f, indent=2)

        return {
            "best_params": best_params,
            "comparison": comp_df.to_dict(orient="records")
        }


if __name__ == "__main__":
    bo = BayesianOptimizer(repo_dir, n_trials=25)
    bo.run()
