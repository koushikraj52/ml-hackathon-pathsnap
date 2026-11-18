"""
Feature engineering and candidate generation for PathSnap.
Extracts spatial, directional, topological, and kinematic features.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional
from src.data_loader import project_point_to_segment, compute_bearing, heading_difference


FEATURE_NAMES = [
    "dist_to_segment",
    "heading_diff",
    "cos_heading_diff",
    "proj_ratio",
    "arc_length",
    "dist_to_u",
    "dist_to_v",
    "gps_speed",
    "dist_rank",
    "emission_prob"
]


class FeatureExtractor:
    def __init__(self, sigma_z: float = 4.07, top_k_candidates: int = 10):
        self.sigma_z = sigma_z
        self.top_k = top_k_candidates

    def build_spatial_index(self, nodes: np.ndarray, arcs: np.ndarray, kx: float, ky: float):
        """
        Builds KDTree on road network nodes and mapping from node to incident arcs.
        """
        node_to_arcs: Dict[int, List[int]] = {}
        for aidx, (u, v) in enumerate(arcs):
            node_to_arcs.setdefault(u, []).append(aidx)
            node_to_arcs.setdefault(v, []).append(aidx)

        nodes_metric = np.column_stack([nodes[:, 0] * kx, nodes[:, 1] * ky])
        tree = cKDTree(nodes_metric)
        return tree, node_to_arcs

    def project_trajectory_to_route(
        self,
        track: np.ndarray,
        nodes: np.ndarray,
        arcs: np.ndarray,
        route: List[int],
        kx: float,
        ky: float
    ) -> List[int]:
        """
        Maps each GPS observation to its corresponding ground-truth route arc
        by sequentially advancing along the ground-truth route sequence.
        """
        if not route:
            return []

        arc_idx = 0
        gt_arcs = []
        n_route = len(route)

        for i in range(len(track)):
            p = track[i, :2]
            best_d = float('inf')
            best_aidx = arc_idx
            best_arc = route[arc_idx]

            # Monotonic sequential search window along route
            w_start = max(0, arc_idx - 1)
            w_end = min(n_route, arc_idx + 20)

            for aidx in range(w_start, w_end):
                arc_id = route[aidx]
                u, v = arcs[arc_id]
                _, d, _ = project_point_to_segment(
                    p[0], p[1],
                    nodes[u, 0], nodes[u, 1],
                    nodes[v, 0], nodes[v, 1],
                    kx, ky
                )
                if d < best_d:
                    best_d = d
                    best_aidx = aidx
                    best_arc = arc_id

            arc_idx = best_aidx
            gt_arcs.append(best_arc)

        return gt_arcs

    def extract_candidates_for_observation(
        self,
        obs_idx: int,
        track: np.ndarray,
        nodes: np.ndarray,
        arcs: np.ndarray,
        node_tree: cKDTree,
        node_to_arcs: Dict[int, List[int]],
        kx: float,
        ky: float,
        target_arc: Optional[int] = None
    ) -> List[Dict]:
        """
        Extracts candidate arcs and features for one GPS observation.
        """
        pt = track[obs_idx]
        px, py, ts = pt[0], pt[1], pt[2]

        # GPS bearing from adjacent observations
        if obs_idx == 0 and len(track) > 1:
            next_pt = track[1]
            gps_bearing = compute_bearing(px, py, next_pt[0], next_pt[1])
            dt = max(1e-3, next_pt[2] - ts)
            speed = np.sqrt(((next_pt[0] - px) * kx)**2 + ((next_pt[1] - py) * ky)**2) / dt
        elif obs_idx > 0:
            prev_pt = track[obs_idx - 1]
            gps_bearing = compute_bearing(prev_pt[0], prev_pt[1], px, py)
            dt = max(1e-3, ts - prev_pt[2])
            speed = np.sqrt(((px - prev_pt[0]) * kx)**2 + ((py - prev_pt[1]) * ky)**2) / dt
        else:
            gps_bearing = 0.0
            speed = 0.0

        # Query KDTree for nearest nodes (k=12)
        pt_metric = np.array([px * kx, py * ky])
        k_query = min(12, len(nodes))
        _, nearest_nids = node_tree.query(pt_metric, k=k_query)
        if isinstance(nearest_nids, (int, np.integer)):
            nearest_nids = [nearest_nids]

        cand_arcs = set()
        for nid in nearest_nids:
            for aidx in node_to_arcs.get(int(nid), []):
                cand_arcs.add(aidx)

        # Ensure target arc is in candidate set during training for evaluation of candidate ranking
        if target_arc is not None:
            cand_arcs.add(target_arc)

        candidates = []
        for aidx in cand_arcs:
            u, v = arcs[aidx]
            u_pt = nodes[u]
            v_pt = nodes[v]

            t_proj, dist_seg, seg_len = project_point_to_segment(
                px, py, u_pt[0], u_pt[1], v_pt[0], v_pt[1], kx, ky
            )

            # Arc bearing
            arc_bearing = compute_bearing(u_pt[0], u_pt[1], v_pt[0], v_pt[1])
            h_diff = heading_difference(gps_bearing, arc_bearing)
            cos_h_diff = float(np.cos(np.radians(h_diff)))

            # Distance to endpoints
            d_u = float(np.sqrt(((px - u_pt[0]) * kx)**2 + ((py - u_pt[1]) * ky)**2))
            d_v = float(np.sqrt(((px - v_pt[0]) * kx)**2 + ((py - v_pt[1]) * ky)**2))

            # Gaussian emission prob
            emission = float(np.exp(-0.5 * (dist_seg / self.sigma_z)**2))

            is_target = 1 if (target_arc is not None and aidx == target_arc) else 0

            candidates.append({
                "arc_id": aidx,
                "dist_to_segment": dist_seg,
                "heading_diff": h_diff,
                "cos_heading_diff": cos_h_diff,
                "proj_ratio": t_proj,
                "arc_length": seg_len,
                "dist_to_u": d_u,
                "dist_to_v": d_v,
                "gps_speed": speed,
                "emission_prob": emission,
                "label": is_target
            })

        # Sort candidates by distance to segment
        candidates.sort(key=lambda c: c["dist_to_segment"])

        # Retain top K candidates
        # If target_arc is present but beyond top_k, retain it to keep positive class
        top_candidates = candidates[:self.top_k]
        if target_arc is not None and not any(c["label"] == 1 for c in top_candidates):
            for c in candidates[self.top_k:]:
                if c["label"] == 1:
                    top_candidates.append(c)
                    break

        n_cands = len(top_candidates)
        for rank, c in enumerate(top_candidates):
            c["dist_rank"] = float(rank / max(1, n_cands - 1))

        return top_candidates

    def extract_trip_features(self, trip_data: Dict, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Extracts features and labels for all observations in a trip.
        Returns:
          X: (N_cand, n_features)
          y: (N_cand,)
          metadata: list of dicts with obs_idx, arc_id, etc.
        """
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        route = trip_data["route"]
        kx, ky = trip_data["kx"], trip_data["ky"]

        node_tree, node_to_arcs = self.build_spatial_index(nodes, arcs, kx, ky)

        gt_arcs = []
        if is_training and route:
            gt_arcs = self.project_trajectory_to_route(track, nodes, arcs, route, kx, ky)

        all_X = []
        all_y = []
        metadata = []

        for obs_i in range(len(track)):
            target = gt_arcs[obs_i] if gt_arcs else None
            cands = self.extract_candidates_for_observation(
                obs_i, track, nodes, arcs, node_tree, node_to_arcs, kx, ky, target_arc=target
            )

            for c in cands:
                feat = [c[f] for f in FEATURE_NAMES]
                all_X.append(feat)
                all_y.append(c["label"])
                metadata.append({
                    "obs_idx": obs_i,
                    "arc_id": c["arc_id"],
                    "dist_to_segment": c["dist_to_segment"],
                    "emission_prob": c["emission_prob"],
                    "label": c["label"]
                })

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int32)
        return X, y, metadata
