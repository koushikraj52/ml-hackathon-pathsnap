"""
Plotting and visual analytics module for PathSnap.
Generates all required diagnostic charts, EDA figures, model comparisons, and loss curves.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution


class PlotGenerator:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.figures_dir = os.path.join(repo_dir, "artifacts", "figures")
        self.results_dir = os.path.join(repo_dir, "artifacts", "results")
        os.makedirs(self.figures_dir, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "Arial"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_eda(self, summary_df: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Trajectory Lengths
        ax1 = axes[0, 0]
        ax1.hist(summary_df["num_track_points"], bins=15, color="#2563eb", edgecolor="black", alpha=0.8)
        ax1.set_title("Distribution of GPS Trajectory Lengths", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Number of GPS Observations per Trip")
        ax1.set_ylabel("Frequency (Trips)")

        # 2. Road Network Size
        ax2 = axes[0, 1]
        ax2.scatter(summary_df["num_nodes"], summary_df["num_arcs"], color="#059669", alpha=0.7, edgecolors="none", s=50)
        ax2.set_title("Road Network Nodes vs Arcs per Trip Corridor", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Number of Nodes")
        ax2.set_ylabel("Number of Arcs")

        # 3. Sampling Rates
        ax3 = axes[1, 0]
        ax3.hist(summary_df["pct_one_hz"], bins=10, color="#d97706", edgecolor="black", alpha=0.8)
        ax3.set_title("Nominal 1 Hz Sampling Fidelity", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Percentage of 1.0s Interval Steps (%)")
        ax3.set_ylabel("Frequency (Trips)")

        # 4. Speeds
        ax4 = axes[1, 1]
        ax4.hist(summary_df["avg_speed_kmh"], bins=15, color="#7c3aed", edgecolor="black", alpha=0.8)
        ax4.set_title("Average Vehicle Speed Distribution", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Average Speed (km/h)")
        ax4.set_ylabel("Frequency (Trips)")

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "eda_dataset_distributions.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved EDA plot to {save_path}")

    def plot_model_comparison(self, comp_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. CV vs Student-Test Macro F1
        ax1 = axes[0]
        x = np.arange(len(comp_df))
        w = 0.35
        ax1.bar(x - w/2, comp_df["cv_macro_f1"], width=w, label="CV Macro F1", color="#3b82f6", edgecolor="black")
        ax1.bar(x + w/2, comp_df["test_macro_f1"], width=w, label="Student-Test Macro F1", color="#10b981", edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels(comp_df["model_name"], rotation=20, ha="right")
        ax1.set_ylim(0.5, 1.0)
        ax1.set_title("Model Comparison: CV vs Student-Test Macro F1", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Macro F1-Score")
        ax1.legend(loc="lower right")

        # 2. Top-1 Observation Accuracy & Route Continuity
        ax2 = axes[1]
        ax2.bar(x - w/2, comp_df["test_top1_accuracy"] * 100.0, width=w, label="Top-1 Accuracy (%)", color="#6366f1", edgecolor="black")
        ax2.bar(x + w/2, comp_df["test_route_continuity"] * 100.0, width=w, label="Route Continuity (%)", color="#f59e0b", edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels(comp_df["model_name"], rotation=20, ha="right")
        ax2.set_ylim(40, 105)
        ax2.set_title("Student-Testing Top-1 Accuracy & Route Continuity", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score (%)")
        ax2.legend(loc="lower right")

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "model_comparison_bar.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Model Comparison plot to {save_path}")

    def plot_training_loss_curves(self, models_dict: Dict):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Gradient Boosting score curve
        gb = models_dict.get("Gradient_Boosting")
        if gb and hasattr(gb.model, "train_score_") and len(gb.model.train_score_) > 0:
            axes[0].plot(gb.model.train_score_, label="Training Loss / Pseudo-Score", color="#2563eb", linewidth=2)
            axes[0].set_title("Gradient Boosting Training History", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("Iteration / Tree")
            axes[0].set_ylabel("Training Score")
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, "Gradient Boosting Loss Curve", ha="center")

        # MLP loss curve
        mlp = models_dict.get("MLP_Neural_Net")
        if mlp and hasattr(mlp.model, "loss_curve_") and len(mlp.model.loss_curve_) > 0:
            axes[1].plot(mlp.model.loss_curve_, label="Cross-Entropy Training Loss", color="#dc2626", linewidth=2)
            axes[1].set_title("MLP Neural Network Training Loss Curve", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Epoch / Iteration")
            axes[1].set_ylabel("Log Loss")
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "MLP Loss Curve", ha="center")

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "training_loss_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Training Loss Curves to {save_path}")

    def plot_bayesian_optimization(self, trials_df: pd.DataFrame):
        plt.figure(figsize=(9, 5))
        valid_trials = trials_df[trials_df["value"].notnull()]
        cum_max = valid_trials["value"].cummax()

        plt.plot(valid_trials["number"], valid_trials["value"], "o--", color="#94a3b8", alpha=0.7, label="Trial Objective (CV F1)")
        plt.plot(valid_trials["number"], cum_max, "-", color="#0284c7", linewidth=2.5, label="Best Cumulative Objective")

        plt.title("Bayesian Hyperparameter Optimization (Optuna TPE)", fontsize=13, fontweight="bold")
        plt.xlabel("Trial Number")
        plt.ylabel("Group-Aware CV Macro F1")
        plt.legend(loc="lower right")
        plt.tight_layout()

        save_path = os.path.join(self.figures_dir, "bayesian_opt_history.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved Bayesian Optimization History to {save_path}")

    def plot_hmm_improvement(self, hmm_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 5))
        trips = [f"Trip {tid[-2:]}" for tid in hmm_df["trip_id"]]
        x = np.arange(len(trips))
        w = 0.35

        ax.bar(x - w/2, hmm_df["classifier_continuity"] * 100.0, width=w, label="Classifier-Only Continuity", color="#ef4444", edgecolor="black")
        ax.bar(x + w/2, hmm_df["hmm_viterbi_continuity"] * 100.0, width=w, label="Classifier + HMM Viterbi Continuity", color="#10b981", edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(trips, rotation=30, ha="right")
        ax.set_ylim(40, 105)
        ax.set_ylabel("Route Continuity (%)")
        ax.set_title("Route Continuity Gain: Classifier-Only vs Classifier + HMM Viterbi Decoding", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "hmm_route_continuity_gain.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved HMM Improvement plot to {save_path}")
