"""
Evaluation metrics for PathSnap candidate classification and sequence route matching.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes standard candidate-level classification metrics.
    """
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="binary", zero_division=0))

    try:
        eps = 1e-15
        p1 = np.clip(y_prob[:, 1], eps, 1.0 - eps)
        ll = float(log_loss(y_true, p1))
    except Exception:
        ll = 0.0

    return {
        "macro_f1": f1_macro,
        "binary_f1": f1_binary,
        "precision": precision,
        "recall": recall,
        "log_loss": ll
    }


def evaluate_trip_top1(
    metadata: List[Dict],
    y_prob: np.ndarray,
    arcs: np.ndarray,
    gt_route: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluates observation-level top-1 accuracy and route continuity for a trip.
    """
    # Group candidate predictions by obs_idx
    obs_groups: Dict[int, List[Tuple[float, int, int]]] = {}
    for i, meta in enumerate(metadata):
        obs_idx = meta["obs_idx"]
        prob = float(y_prob[i, 1])
        arc_id = meta["arc_id"]
        label = meta["label"]
        obs_groups.setdefault(obs_idx, []).append((prob, arc_id, label))

    total_obs = len(obs_groups)
    top1_hits = 0
    candidate_misses = 0
    top1_predicted_arcs = []
    ground_truth_arcs = []

    for obs_idx in sorted(obs_groups.keys()):
        cands = obs_groups[obs_idx]
        # Check if GT arc was in candidates
        has_gt = any(c[2] == 1 for c in cands)
        if not has_gt:
            candidate_misses += 1

        # Select candidate with highest predicted probability
        best_cand = max(cands, key=lambda c: c[0])
        top1_predicted_arcs.append(best_cand[1])

        # Find GT label
        gt_cand = [c for c in cands if c[2] == 1]
        gt_arc = gt_cand[0][1] if gt_cand else None
        ground_truth_arcs.append(gt_arc)

        if best_cand[2] == 1:
            top1_hits += 1

    top1_acc = float(top1_hits / total_obs) if total_obs > 0 else 0.0
    miss_rate = float(candidate_misses / total_obs) if total_obs > 0 else 0.0

    # Route continuity: fraction of consecutive transitions that are connected
    connected_transitions = 0
    total_transitions = max(1, len(top1_predicted_arcs) - 1)

    for i in range(len(top1_predicted_arcs) - 1):
        arc_curr = top1_predicted_arcs[i]
        arc_next = top1_predicted_arcs[i + 1]
        if arc_curr == arc_next:
            # Staying on the same road segment is valid continuous traversal
            connected_transitions += 1
        else:
            u_curr, v_curr = arcs[arc_curr]
            u_next, v_next = arcs[arc_next]
            if v_curr == u_next:
                connected_transitions += 1

    route_continuity = float(connected_transitions / total_transitions)

    return {
        "total_observations": total_obs,
        "top1_accuracy": top1_acc,
        "candidate_miss_rate": miss_rate,
        "route_continuity": route_continuity,
        "predicted_arcs": top1_predicted_arcs,
        "ground_truth_arcs": ground_truth_arcs
    }
