import sys
import os
repo_root = r"C:/Users/koush/.gemini/antigravity/scratch/ml-hackathon-pathsnap/repo"
sys.path.insert(0, repo_root)

import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.api.app import predict_and_evaluate_trip

def main():
    raw_dir = os.path.join(repo_root, "data", "raw")
    loader = DataLoader(raw_dir)

    eval_trips = [
        "00000015", "00000017", "00000023", "00000030", "00000049",
        "00000059", "00000067", "00000073", "00000087", "00000094"
    ]

    eda_csv = os.path.join(repo_root, "artifacts", "results", "eda_trip_summary.csv")
    df_eda = pd.read_csv(eda_csv, dtype={"trip_id": str})

    new_rows = []
    for tid in eval_trips:
        if tid not in df_eda["trip_id"].values:
            trip = loader.load_trip(tid)
            new_rows.append({
                "trip_id": tid,
                "num_track_points": len(trip["track"]),
                "num_nodes": len(trip["nodes"]),
                "num_arcs": len(trip["arcs"]),
                "num_route_arcs": len(trip["route"]),
                "split": "evaluator_test"
            })

    if new_rows:
        df_eda = pd.concat([df_eda, pd.DataFrame(new_rows)], ignore_index=True)
        df_eda.to_csv(eda_csv, index=False)
        print(f"Added {len(new_rows)} evaluator trips to eda_trip_summary.csv")

    eval_results = []
    for tid in eval_trips:
        res = predict_and_evaluate_trip(tid)
        m = res["metrics"]
        eval_results.append({
            "trip_id": tid,
            "obs_acc_classifier": m["classifier_top1_accuracy"],
            "obs_acc_hmm": m["hmm_viterbi_accuracy"],
            "acc_gain_pct": m["accuracy_gain_pct"],
            "continuity_classifier": m["classifier_continuity"],
            "continuity_hmm": m["hmm_viterbi_continuity"],
            "continuity_gain_pct": m["continuity_gain_pct"],
            "candidate_miss_rate": m["candidate_miss_rate"],
            "total_latency_ms": res["latency"]["total_latency_ms"],
            "ml_inference_ms": res["latency"]["ml_inference_ms"],
            "hmm_viterbi_ms": res["latency"]["hmm_viterbi_ms"]
        })
        print(f"Trip {tid}: Clf Acc={m['classifier_top1_accuracy']}% -> HMM Acc={m['hmm_viterbi_accuracy']}% | HMM Continuity={m['hmm_viterbi_continuity']}%")

    df_eval = pd.DataFrame(eval_results)
    eval_res_csv = os.path.join(repo_root, "artifacts", "results", "faculty_evaluator_test_results.csv")
    df_eval.to_csv(eval_res_csv, index=False)

    print("\n================ OFFICIAL FACULTY TEST SET EVALUATION ================")
    print(f"Mean Standalone Classifier Accuracy: {df_eval['obs_acc_classifier'].mean():.2f}%")
    print(f"Mean HMM Viterbi Accuracy:           {df_eval['obs_acc_hmm'].mean():.2f}%")
    print(f"Mean HMM Route Continuity:           {df_eval['continuity_hmm'].mean():.2f}%")
    print(f"Mean Total Pipeline Latency:         {df_eval['total_latency_ms'].mean():.2f} ms")
    print("=======================================================================")

if __name__ == "__main__":
    main()
