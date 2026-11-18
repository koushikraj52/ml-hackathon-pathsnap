"""
Benchmark and comparison between Classifier-Only vs Classifier + HMM Viterbi decoding.
"""

import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.hmm.graph import RoadNetworkGraph
from src.hmm.viterbi import HMMViterbiDecoder
from src.visualization.plots import PlotGenerator


class HMMBenchmark:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.results_dir = os.path.join(repo_dir, "artifacts", "results")
        self.model_path = os.path.join(repo_dir, "artifacts", "models", "best_model_tuned.joblib")
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join(repo_dir, "artifacts", "models", "Gradient_Boosting.joblib")
        self.test_path = os.path.join(repo_dir, "data", "processed", "student_test_features.joblib")
        self.plot_gen = PlotGenerator(repo_dir)

    def run(self) -> pd.DataFrame:
        print("\n=== Running HMM / Viterbi Route Reconstruction Benchmark ===")
        model = joblib.load(self.model_path)
        test_data = joblib.load(self.test_path)

        X_test = test_data["X"]
        y_probs = model.predict_proba(X_test)

        records = []
        offset = 0

        for tid in test_data["trip_ids"]:
            t_info = test_data["trip_metadata"][tid]
            n_samples = t_info["n_samples"]
            t_probs = y_probs[offset : offset + n_samples]
            t_meta = t_info["metadata"]
            t_arcs = t_info["arcs"]
            t_nodes = t_info["nodes"]
            t_track = t_info["track"]
            kx, ky = t_info["kx"], t_info["ky"]

            # 1. Classifier-Only Top-1
            obs_groups: Dict[int, List[Dict]] = {}
            for i, meta in enumerate(t_meta):
                obs_idx = meta["obs_idx"]
                c_dict = dict(meta)
                c_dict["model_prob"] = float(t_probs[i, 1])
                obs_groups.setdefault(obs_idx, []).append(c_dict)

            classifier_pred_arcs = []
            gt_arcs = []
            for obs_idx in sorted(obs_groups.keys()):
                cands = obs_groups[obs_idx]
                best_c = max(cands, key=lambda c: c["model_prob"])
                classifier_pred_arcs.append(best_c["arc_id"])
                gt_c = [c for c in cands if c["label"] == 1]
                gt_arcs.append(gt_c[0]["arc_id"] if gt_c else -1)

            # 2. HMM + Viterbi Decoding
            graph = RoadNetworkGraph(t_nodes, t_arcs, kx, ky)
            decoder = HMMViterbiDecoder(
                graph=graph,
                beta=12.0,
                stay_log_bonus=1.2,
                direct_trans_log_prob=0.0,
                disconnect_penalty=-15.0
            )

            cand_lists = [obs_groups[k] for k in sorted(obs_groups.keys())]
            viterbi_pred_arcs = decoder.decode(cand_lists, t_track, kx, ky)

            n_obs = len(gt_arcs)
            clf_hits = sum(1 for p, g in zip(classifier_pred_arcs, gt_arcs) if p == g)
            vit_hits = sum(1 for p, g in zip(viterbi_pred_arcs, gt_arcs) if p == g)

            clf_acc = float(clf_hits / n_obs)
            vit_acc = float(vit_hits / n_obs)

            def calc_continuity(pred_list):
                connected = 0
                total = max(1, len(pred_list) - 1)
                for i in range(len(pred_list) - 1):
                    a1 = pred_list[i]
                    a2 = pred_list[i + 1]
                    if a1 == a2:
                        connected += 1
                    else:
                        u1, v1 = t_arcs[a1]
                        u2, v2 = t_arcs[a2]
                        if v1 == u2:
                            connected += 1
                return float(connected / total)

            clf_cont = calc_continuity(classifier_pred_arcs)
            vit_cont = calc_continuity(viterbi_pred_arcs)

            records.append({
                "trip_id": tid,
                "observations": n_obs,
                "classifier_top1_acc": clf_acc,
                "hmm_viterbi_top1_acc": vit_acc,
                "acc_gain": vit_acc - clf_acc,
                "classifier_continuity": clf_cont,
                "hmm_viterbi_continuity": vit_cont,
                "continuity_gain": vit_cont - clf_cont
            })

            offset += n_samples

        df_bench = pd.DataFrame(records)
        bench_csv = os.path.join(self.results_dir, "hmm_vs_classifier.csv")
        df_bench.to_csv(bench_csv, index=False)

        print(f"\nSaved HMM Benchmark to {bench_csv}")
        print(df_bench.to_string())

        # Generate HMM Improvement figure
        self.plot_gen.plot_hmm_improvement(df_bench)

        summary = {
            "Mean Classifier Top-1 Accuracy": float(df_bench["classifier_top1_acc"].mean()),
            "Mean HMM Viterbi Top-1 Accuracy": float(df_bench["hmm_viterbi_top1_acc"].mean()),
            "Mean Classifier Continuity": float(df_bench["classifier_continuity"].mean()),
            "Mean HMM Viterbi Continuity": float(df_bench["hmm_viterbi_continuity"].mean()),
        }
        print("\nOverall Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v*100:.2f}%")

        with open(os.path.join(self.results_dir, "hmm_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        return df_bench


if __name__ == "__main__":
    bench = HMMBenchmark(repo_dir)
    bench.run()
