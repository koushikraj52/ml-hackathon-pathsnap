"""
Model experimentation and comparison pipeline for PathSnap.
Trains and compares 5 models across distinct algorithm families using GroupKFold CV.
"""

import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss

from src.data_loader import DataLoader
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.models import get_all_models, BaseModel
from src.evaluation.metrics import compute_classification_metrics, evaluate_trip_top1
from src.visualization.plots import PlotGenerator


class TrainingPipeline:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.data_loader = DataLoader(os.path.join(repo_dir, "data", "raw"))
        self.feature_extractor = FeatureExtractor(sigma_z=4.07, top_k_candidates=10)
        self.splits_file = os.path.join(repo_dir, "data", "splits", "splits.json")
        self.cache_dir = os.path.join(repo_dir, "data", "processed")
        self.results_dir = os.path.join(repo_dir, "artifacts", "results")
        self.models_dir = os.path.join(repo_dir, "artifacts", "models")
        self.plot_gen = PlotGenerator(repo_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        with open(self.splits_file, "r") as f:
            self.splits = json.load(f)

        self.train_trip_ids = self.splits["train_trips"]
        self.student_test_ids = self.splits["student_test_trips"]

    def extract_or_load_dataset(self, trip_ids: List[str], partition_name: str) -> Dict[str, Any]:
        cache_path = os.path.join(self.cache_dir, f"{partition_name}_features.joblib")
        if os.path.exists(cache_path):
            print(f"Loading cached {partition_name} dataset from {cache_path}...")
            return joblib.load(cache_path)

        print(f"Extracting features for {partition_name} ({len(trip_ids)} trips)...")
        all_X = []
        all_y = []
        all_groups = []
        trip_metadata = {}

        t0 = time.time()
        for idx, tid in enumerate(trip_ids):
            t_data = self.data_loader.load_trip(tid)
            X, y, meta = self.feature_extractor.extract_trip_features(t_data, is_training=True)

            all_X.append(X)
            all_y.append(y)
            all_groups.extend([idx] * len(y))
            trip_metadata[tid] = {
                "metadata": meta,
                "track": t_data["track"],
                "arcs": t_data["arcs"],
                "nodes": t_data["nodes"],
                "route": t_data["route"],
                "kx": t_data["kx"],
                "ky": t_data["ky"],
                "n_samples": len(y)
            }

            if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_ids):
                print(f"  Processed {idx + 1}/{len(trip_ids)} trips (elapsed: {time.time() - t0:.1f}s)")

        X_cat = np.vstack(all_X)
        y_cat = np.concatenate(all_y)
        groups_cat = np.array(all_groups, dtype=int)

        dataset = {
            "X": X_cat,
            "y": y_cat,
            "groups": groups_cat,
            "trip_ids": trip_ids,
            "trip_metadata": trip_metadata
        }

        joblib.dump(dataset, cache_path, compress=3)
        print(f"Cached {partition_name}: X shape {X_cat.shape}, y balance {np.bincount(y_cat)}")
        return dataset

    def run_cross_validation(self, models: Dict[str, BaseModel], train_data: Dict) -> Dict[str, float]:
        print("\n--- Running 5-Fold GroupKFold Cross-Validation ---")
        X = train_data["X"]
        y = train_data["y"]
        groups = train_data["groups"]

        # Subsample for CV efficiency
        sub_indices = np.arange(0, len(X), max(1, len(X) // 35000))
        X_sub, y_sub, groups_sub = X[sub_indices], y[sub_indices], groups[sub_indices]

        gkf = GroupKFold(n_splits=5)
        cv_scores = {name: [] for name in models}

        fold = 0
        for train_idx, val_idx in gkf.split(X_sub, y_sub, groups_sub):
            fold += 1
            X_tr, y_tr = X_sub[train_idx], y_sub[train_idx]
            X_val, y_val = X_sub[val_idx], y_sub[val_idx]

            for name in models:
                fresh_models = get_all_models()
                m = fresh_models[name]
                m.fit(X_tr, y_tr)
                preds = m.predict(X_val)
                f1_macro = f1_score(y_val, preds, average="macro", zero_division=0)
                cv_scores[name].append(f1_macro)

        mean_cv = {name: float(np.mean(cv_scores[name])) for name in models}
        for name, score in mean_cv.items():
            print(f"  {name}: Mean CV Macro F1 = {score:.4f} (+/- {np.std(cv_scores[name]):.4f})")
        return mean_cv

    def evaluate_on_student_test(
        self,
        model_name: str,
        model: BaseModel,
        test_data: Dict
    ) -> Dict[str, Any]:
        X_test = test_data["X"]
        y_test = test_data["y"]

        t0 = time.time()
        y_probs = model.predict_proba(X_test)
        inference_time_ms = ((time.time() - t0) / max(1, len(X_test))) * 1000.0

        y_preds = np.argmax(y_probs, axis=1)
        class_metrics = compute_classification_metrics(y_test, y_preds, y_probs)

        top1_accs = []
        route_continuities = []
        miss_rates = []

        curr_offset = 0
        for tid in test_data["trip_ids"]:
            t_info = test_data["trip_metadata"][tid]
            n_samples = t_info["n_samples"]
            t_probs = y_probs[curr_offset : curr_offset + n_samples]
            t_meta = t_info["metadata"]
            t_arcs = t_info["arcs"]

            res = evaluate_trip_top1(t_meta, t_probs, t_arcs)
            top1_accs.append(res["top1_accuracy"])
            route_continuities.append(res["route_continuity"])
            miss_rates.append(res["candidate_miss_rate"])

            curr_offset += n_samples

        return {
            "model_name": model.name,
            "algorithm_family": model.algorithm_family,
            "test_macro_f1": class_metrics["macro_f1"],
            "test_binary_f1": class_metrics["binary_f1"],
            "test_precision": class_metrics["precision"],
            "test_recall": class_metrics["recall"],
            "test_log_loss": class_metrics["log_loss"],
            "test_top1_accuracy": float(np.mean(top1_accs)),
            "test_route_continuity": float(np.mean(route_continuities)),
            "test_candidate_miss_rate": float(np.mean(miss_rates)),
            "inference_time_ms": inference_time_ms,
            "training_time_sec": model.training_time_sec
        }

    def run(self) -> pd.DataFrame:
        train_data = self.extract_or_load_dataset(self.train_trip_ids, "train")
        test_data = self.extract_or_load_dataset(self.student_test_ids, "student_test")

        models = get_all_models()

        # Cross-validation
        cv_scores = self.run_cross_validation(models, train_data)

        # Full training on train_data and evaluation on student_test
        print("\n--- Training on Full Training Set and Evaluating on Student Test ---")
        records = []
        fitted_models = {}

        # Subsample training data to 50k points if massive to keep training snappy and accurate
        X_train_full = train_data["X"]
        y_train_full = train_data["y"]
        if len(X_train_full) > 60000:
            sub_tr = np.arange(0, len(X_train_full), max(1, len(X_train_full) // 50000))
            X_train_fit = X_train_full[sub_tr]
            y_train_fit = y_train_full[sub_tr]
        else:
            X_train_fit = X_train_full
            y_train_fit = y_train_full

        for name, model in models.items():
            print(f"Fitting {name} ({model.algorithm_family}) on {len(X_train_fit)} samples...")
            model.fit(X_train_fit, y_train_fit)
            fitted_models[name] = model

            # Save model artifact
            model_save_path = os.path.join(self.models_dir, f"{name}.joblib")
            joblib.dump(model, model_save_path)

            eval_res = self.evaluate_on_student_test(name, model, test_data)
            eval_res["cv_macro_f1"] = cv_scores[name]
            records.append(eval_res)
            print(f"  -> Top-1 Acc: {eval_res['test_top1_accuracy']*100:.2f}%, Macro F1: {eval_res['test_macro_f1']:.4f}, Continuity: {eval_res['test_route_continuity']*100:.2f}%")

        df_results = pd.DataFrame(records)
        cols_order = [
            "model_name", "algorithm_family", "cv_macro_f1", "test_top1_accuracy",
            "test_route_continuity", "test_macro_f1", "test_log_loss",
            "test_precision", "test_recall", "inference_time_ms", "training_time_sec"
        ]
        df_results = df_results[cols_order]

        csv_path = os.path.join(self.results_dir, "model_comparison.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nSaved model comparison table to {csv_path}")
        print(df_results.to_string())

        # Generate comparison charts & loss curves
        self.plot_gen.plot_model_comparison(df_results)
        self.plot_gen.plot_training_loss_curves(fitted_models)

        return df_results


if __name__ == "__main__":
    pipeline = TrainingPipeline(repo_dir)
    pipeline.run()
