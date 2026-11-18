"""
FastAPI Backend for PathSnap Distributed Map-Matching & Route Reconstruction.
Implements Laptop 1 server responsibilities: model inference, HMM decoding,
server-side accuracy evaluation, and interactive Folium map serving.
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Any, Optional

from src.data_loader import DataLoader
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.hmm.graph import RoadNetworkGraph
from src.hmm.viterbi import HMMViterbiDecoder
from src.visualization.folium_maps import FoliumMapGenerator


app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed ML and HMM-based route reconstruction service for GPS trajectories",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_DATA_DIR = os.path.join(repo_dir, "data", "raw")
SPLITS_FILE = os.path.join(repo_dir, "data", "splits", "splits.json")
MODELS_DIR = os.path.join(repo_dir, "artifacts", "models")
MAPS_DIR = os.path.join(repo_dir, "artifacts", "maps")
UI_DIR = os.path.join(repo_dir, "src", "ui")

data_loader = DataLoader(RAW_DATA_DIR)
feature_extractor = FeatureExtractor(sigma_z=4.07, top_k_candidates=10)
map_generator = FoliumMapGenerator(repo_dir)

evaluation_trip_ids = []
if os.path.exists(SPLITS_FILE):
    with open(SPLITS_FILE, "r") as f:
        splits = json.load(f)
        evaluation_trip_ids = splits.get("evaluator_test_trips", [])
if not evaluation_trip_ids:
    evaluation_trip_ids = data_loader.list_available_trips()[:10]

# Fast metadata loading from eda_trip_summary.csv or on-demand
trip_metadata_cache = {}
eda_csv = os.path.join(repo_dir, "artifacts", "results", "eda_trip_summary.csv")
if os.path.exists(eda_csv):
    try:
        import pandas as pd
        df_eda = pd.read_csv(eda_csv)
        for _, row in df_eda.iterrows():
            tid = str(row["trip_id"]).zfill(8)
            trip_metadata_cache[tid] = {
                "trip_id": tid,
                "num_track_points": int(row["num_track_points"]),
                "num_nodes": int(row["num_nodes"]),
                "num_arcs": int(row["num_arcs"]),
                "has_reference_route": int(row.get("num_route_arcs", 0)) > 0,
                "duration_seconds": float(row["num_track_points"])
            }
    except Exception:
        pass

best_model = None
model_name_loaded = "None"
possible_models = [
    os.path.join(MODELS_DIR, "best_model_tuned.joblib"),
    os.path.join(MODELS_DIR, "Gradient_Boosting.joblib"),
    os.path.join(MODELS_DIR, "Random_Forest.joblib"),
    os.path.join(MODELS_DIR, "Decision_Tree.joblib"),
    os.path.join(MODELS_DIR, "Logistic_Regression.joblib")
]
for p in possible_models:
    if os.path.exists(p):
        best_model = joblib.load(p)
        model_name_loaded = os.path.basename(p)
        break


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PathSnap ML & HMM Backend",
        "model_loaded": model_name_loaded,
        "evaluation_trips_count": len(evaluation_trip_ids),
        "timestamp": time.time()
    }


@app.get("/api/trips")
def get_evaluation_trips():
    return {"evaluation_trips": [trip_metadata_cache[tid] for tid in evaluation_trip_ids if tid in trip_metadata_cache]}


@app.post("/api/predict/{trip_id}")
def predict_and_evaluate_trip(trip_id: str):
    global best_model
    if best_model is None:
        raise HTTPException(status_code=500, detail="No trained model available.")

    try:
        trip_data = data_loader.load_trip(trip_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found: {str(e)}")

    track = trip_data["track"]
    nodes = trip_data["nodes"]
    arcs = trip_data["arcs"]
    route = trip_data["route"]
    kx, ky = trip_data["kx"], trip_data["ky"]

    t_start = time.time()

    # Step 1: Feature Extraction
    X_cand, _, meta = feature_extractor.extract_trip_features(trip_data, is_training=False)

    # Step 2: Model Inference
    t_inf_start = time.time()
    y_probs = best_model.predict_proba(X_cand)
    inf_duration_ms = (time.time() - t_inf_start) * 1000.0

    # Step 3: Classifier Top-1
    obs_groups: Dict[int, List[Dict]] = {}
    for i, m in enumerate(meta):
        c_dict = dict(m)
        c_dict["model_prob"] = float(y_probs[i, 1])
        obs_groups.setdefault(m["obs_idx"], []).append(c_dict)

    classifier_pred_arcs = []
    for obs_i in range(len(track)):
        cands = obs_groups[obs_i]
        best_c = max(cands, key=lambda c: c["model_prob"])
        classifier_pred_arcs.append(best_c["arc_id"])

    # Step 4: HMM Viterbi Decoding
    t_hmm_start = time.time()
    graph = RoadNetworkGraph(nodes, arcs, kx, ky)
    decoder = HMMViterbiDecoder(
        graph=graph,
        beta=12.0,
        stay_log_bonus=1.2,
        direct_trans_log_prob=0.0,
        disconnect_penalty=-15.0
    )
    cand_lists = [obs_groups[i] for i in range(len(track))]
    viterbi_pred_arcs = decoder.decode(cand_lists, track, kx, ky)
    hmm_duration_ms = (time.time() - t_hmm_start) * 1000.0

    total_latency_ms = (time.time() - t_start) * 1000.0

    # Step 5: Server-side evaluation
    metrics = {
        "has_ground_truth": len(route) > 0,
        "classifier_top1_accuracy": None,
        "hmm_viterbi_accuracy": None,
        "accuracy_gain_pct": None,
        "classifier_continuity": None,
        "hmm_viterbi_continuity": None,
        "continuity_gain_pct": None,
        "candidate_miss_rate": None
    }

    if route:
        gt_arcs = feature_extractor.project_trajectory_to_route(track, nodes, arcs, route, kx, ky)
        n_obs = len(gt_arcs)

        clf_hits = sum(1 for p, g in zip(classifier_pred_arcs, gt_arcs) if p == g)
        vit_hits = sum(1 for p, g in zip(viterbi_pred_arcs, gt_arcs) if p == g)

        clf_acc = float(clf_hits / n_obs)
        vit_acc = float(vit_hits / n_obs)

        def calc_cont(pred_seq):
            connected = 0
            for i in range(len(pred_seq) - 1):
                if pred_seq[i] == pred_seq[i + 1]:
                    connected += 1
                else:
                    if arcs[pred_seq[i], 1] == arcs[pred_seq[i + 1], 0]:
                        connected += 1
            return float(connected / max(1, len(pred_seq) - 1))

        clf_cont = calc_cont(classifier_pred_arcs)
        vit_cont = calc_cont(viterbi_pred_arcs)

        misses = 0
        for obs_i, gt_a in enumerate(gt_arcs):
            c_arcs = [c["arc_id"] for c in obs_groups[obs_i]]
            if gt_a not in c_arcs:
                misses += 1
        miss_rate = float(misses / n_obs)

        metrics.update({
            "classifier_top1_accuracy": round(clf_acc * 100.0, 2),
            "hmm_viterbi_accuracy": round(vit_acc * 100.0, 2),
            "accuracy_gain_pct": round((vit_acc - clf_acc) * 100.0, 2),
            "classifier_continuity": round(clf_cont * 100.0, 2),
            "hmm_viterbi_continuity": round(vit_cont * 100.0, 2),
            "continuity_gain_pct": round((vit_cont - clf_cont) * 100.0, 2),
            "candidate_miss_rate": round(miss_rate * 100.0, 2)
        })

    # Generate Folium Map
    map_filename = f"trip_{trip_id}_evaluated.html"
    map_generator.create_trip_map(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        reference_route_arcs=route if route else None,
        predicted_route_arcs=viterbi_pred_arcs,
        save_filename=map_filename
    )

    return {
        "trip_id": trip_id,
        "observations_count": len(track),
        "model_used": model_name_loaded,
        "metrics": metrics,
        "latency": {
            "total_latency_ms": round(total_latency_ms, 2),
            "ml_inference_ms": round(inf_duration_ms, 2),
            "hmm_viterbi_ms": round(hmm_duration_ms, 2),
            "ms_per_observation": round(total_latency_ms / len(track), 3)
        },
        "map_url": f"/api/map/{trip_id}",
        "predicted_route_arcs_sample": viterbi_pred_arcs[:30]
    }


@app.get("/api/map/{trip_id}", response_class=HTMLResponse)
def get_trip_map_html(trip_id: str):
    map_path = os.path.join(MAPS_DIR, f"trip_{trip_id}_evaluated.html")
    if not os.path.exists(map_path):
        try:
            trip_data = data_loader.load_trip(trip_id)
            map_generator.create_trip_map(
                trip_id=trip_id,
                track=trip_data["track"],
                nodes=trip_data["nodes"],
                arcs=trip_data["arcs"],
                reference_route_arcs=trip_data["route"],
                predicted_route_arcs=None,
                save_filename=f"trip_{trip_id}_evaluated.html"
            )
        except Exception:
            raise HTTPException(status_code=404, detail="Map not found and could not be generated.")

    with open(map_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if os.path.exists(UI_DIR):
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
