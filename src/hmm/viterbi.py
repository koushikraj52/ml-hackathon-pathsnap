"""
Hidden Markov Model (HMM) and Viterbi decoding algorithm for PathSnap route reconstruction.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from src.hmm.graph import RoadNetworkGraph


class HMMViterbiDecoder:
    def __init__(
        self,
        graph: RoadNetworkGraph,
        beta: float = 10.0,
        stay_log_bonus: float = 1.0,
        direct_trans_log_prob: float = 0.0,
        disconnect_penalty: float = -12.0
    ):
        """
        Parameters:
        - beta: scale parameter for difference between network distance and GPS distance
        - stay_log_bonus: bonus for staying on the same arc
        - direct_trans_log_prob: baseline log probability for direct connected transition
        - disconnect_penalty: heavy log penalty for disconnected / illegal transitions
        """
        self.graph = graph
        self.beta = beta
        self.stay_log_bonus = stay_log_bonus
        self.direct_trans_log_prob = direct_trans_log_prob
        self.disconnect_penalty = disconnect_penalty

    def compute_emission_log_probs(self, candidate_probs: List[float]) -> np.ndarray:
        """
        Converts classifier probability predictions into normalized log-emissions.
        """
        probs = np.array(candidate_probs, dtype=float)
        # Softmax or probability floor
        eps = 1e-6
        probs = np.maximum(probs, eps)
        # Normalize across observation candidate set
        p_norm = probs / np.sum(probs)
        return np.log(p_norm)

    def compute_transition_log_prob(
        self,
        prev_arc: int,
        curr_arc: int,
        gps_dist: float,
        dt: float
    ) -> float:
        """
        Computes transition log probability between two candidate arcs.
        Handles:
        1. Staying on same arc
        2. Direct topological transition (u2 == v1)
        3. Multi-hop reachable transition
        4. Disconnected / large GPS gap
        """
        # If large GPS gap (> 5 seconds), soften the penalty
        gap_factor = 0.5 if dt > 5.0 else 1.0

        if prev_arc == curr_arc:
            # Vehicle stayed on the same road segment
            return self.stay_log_bonus

        if self.graph.is_direct_successor(prev_arc, curr_arc):
            # Direct connection in the road network
            return self.direct_trans_log_prob

        # Multi-hop shortest network distance
        net_dist = self.graph.get_shortest_network_distance(prev_arc, curr_arc, max_hops=3)
        if np.isfinite(net_dist):
            dist_diff = abs(net_dist - gps_dist)
            log_prob = - (dist_diff / self.beta)
            return float(log_prob)

        # Disconnected transition
        return self.disconnect_penalty * gap_factor

    def decode(
        self,
        observation_candidates: List[List[Dict]],
        track: np.ndarray,
        kx: float,
        ky: float
    ) -> List[int]:
        """
        Runs Viterbi dynamic programming in log-space across all observations.
        Returns the optimal sequence of predicted arc IDs.
        """
        n_obs = len(observation_candidates)
        if n_obs == 0:
            return []

        # T[t][j]: max log-prob ending at candidate j of observation t
        # Backpointers: B[t][j]: index of candidate at observation t-1 that led to (t, j)
        T: List[np.ndarray] = []
        B: List[np.ndarray] = []

        # Step 0: Initialization
        cands_0 = observation_candidates[0]
        n_cands_0 = len(cands_0)
        probs_0 = [c["model_prob"] for c in cands_0]
        v_0 = self.compute_emission_log_probs(probs_0)
        T.append(v_0)
        B.append(np.zeros(n_cands_0, dtype=int))

        # Steps 1 to T-1: Recurrence
        for t in range(1, n_obs):
            prev_cands = observation_candidates[t - 1]
            curr_cands = observation_candidates[t]
            n_prev = len(prev_cands)
            n_curr = len(curr_cands)

            # GPS step distance and time delta
            dx = (track[t, 0] - track[t - 1, 0]) * kx
            dy = (track[t, 1] - track[t - 1, 1]) * ky
            gps_dist = float(np.sqrt(dx * dx + dy * dy))
            dt = float(max(1e-3, track[t, 2] - track[t - 1, 2]))

            emission_logs = self.compute_emission_log_probs([c["model_prob"] for c in curr_cands])

            v_t = np.zeros(n_curr, dtype=float)
            b_t = np.zeros(n_curr, dtype=int)

            prev_v = T[t - 1]

            for j in range(n_curr):
                curr_arc = curr_cands[j]["arc_id"]
                best_val = -float("inf")
                best_i = 0

                for i in range(n_prev):
                    prev_arc = prev_cands[i]["arc_id"]
                    trans_log = self.compute_transition_log_prob(prev_arc, curr_arc, gps_dist, dt)
                    score = prev_v[i] + trans_log

                    if score > best_val:
                        best_val = score
                        best_i = i

                v_t[j] = best_val + emission_logs[j]
                b_t[j] = best_i

            T.append(v_t)
            B.append(b_t)

        # Backtracking
        best_last_idx = int(np.argmax(T[-1]))
        decoded_arc_sequence = [0] * n_obs
        decoded_arc_sequence[-1] = observation_candidates[-1][best_last_idx]["arc_id"]

        curr_ptr = best_last_idx
        for t in range(n_obs - 1, 0, -1):
            curr_ptr = int(B[t][curr_ptr])
            decoded_arc_sequence[t - 1] = observation_candidates[t - 1][curr_ptr]["arc_id"]

        return decoded_arc_sequence
