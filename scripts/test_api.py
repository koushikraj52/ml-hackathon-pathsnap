"""
Direct Route Handler & API Logic Verification for PathSnap.
Tests health_check, get_evaluation_trips, predict_and_evaluate_trip, and get_trip_map_html.
"""

import sys
import os
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import json
from src.api.app import health_check, get_evaluation_trips, predict_and_evaluate_trip, get_trip_map_html

print("=== Directly Testing API Handlers ===")

# 1. Health check
res_health = health_check()
print(f"1. health_check() -> {res_health}")
assert res_health["status"] == "healthy"

# 2. Evaluation trips
res_trips = get_evaluation_trips()
trips = res_trips["evaluation_trips"]
print(f"2. get_evaluation_trips() -> {len(trips)} evaluation trips returned:")
for t in trips[:3]:
    print(f"   Trip {t['trip_id']}: {t['num_track_points']} points, {t['num_nodes']} nodes, {t['num_arcs']} arcs")
assert len(trips) == 10

# 3. Predict & Evaluate on Trip 00000011
test_trip_id = trips[0]["trip_id"]
print(f"\n3. predict_and_evaluate_trip('{test_trip_id}') ...")
res_pred = predict_and_evaluate_trip(test_trip_id)
print(f"   Observations Count: {res_pred['observations_count']}")
print(f"   Model Used: {res_pred['model_used']}")
print(f"   Total Latency: {res_pred['latency']['total_latency_ms']} ms ({res_pred['latency']['ms_per_observation']} ms/obs)")
print(f"   Metrics: {json.dumps(res_pred['metrics'], indent=2)}")
assert res_pred["metrics"]["hmm_viterbi_accuracy"] > 85.0
assert res_pred["metrics"]["hmm_viterbi_continuity"] > 90.0

# 4. Map HTML
res_map = get_trip_map_html(test_trip_id)
print(f"\n4. get_trip_map_html('{test_trip_id}') -> HTML length: {len(res_map.body)} bytes")
assert len(res_map.body) > 1000

print("\n>>> ALL API LOGIC AND ENDPOINTS VERIFIED SUCCESSFULLY! <<<")
