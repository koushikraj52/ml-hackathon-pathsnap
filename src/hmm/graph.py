"""
Road network graph builder and shortest path utilities for PathSnap HMM transition modeling.
"""

import numpy as np
from typing import Dict, List, Tuple, Set


class RoadNetworkGraph:
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray, kx: float, ky: float):
        self.nodes = nodes
        self.arcs = arcs
        self.kx = kx
        self.ky = ky
        self.num_nodes = len(nodes)
        self.num_arcs = len(arcs)

        # Adjacency list: node_u -> list of (node_v, arc_id, length_meters)
        self.adj: Dict[int, List[Tuple[int, int, float]]] = {}
        # Arc to arc direct transition mapping: arc_i -> set of outgoing arc_ids
        self.arc_successors: Dict[int, Set[int]] = {}

        self._build_graph()

    def _build_graph(self):
        for aidx, (u, v) in enumerate(self.arcs):
            dx = (self.nodes[v, 0] - self.nodes[u, 0]) * self.kx
            dy = (self.nodes[v, 1] - self.nodes[u, 1]) * self.ky
            length = float(np.sqrt(dx * dx + dy * dy))

            self.adj.setdefault(int(u), []).append((int(v), aidx, length))
            self.arc_successors.setdefault(aidx, set())

        # Precompute direct arc successors: to_node of arc_i == from_node of arc_j
        for aidx, (u, v) in enumerate(self.arcs):
            out_edges = self.adj.get(int(v), [])
            for _, next_arc, _ in out_edges:
                self.arc_successors[aidx].add(next_arc)

    def is_direct_successor(self, arc_from: int, arc_to: int) -> bool:
        return arc_to in self.arc_successors.get(arc_from, set())

    def get_shortest_network_distance(
        self, arc_from: int, arc_to: int, max_hops: int = 3
    ) -> float:
        """
        Computes network distance between the end of arc_from and the start of arc_to
        up to max_hops search depth using Breadth-First / Dijkstra search.
        Returns distance in meters, or float('inf') if disconnected within max_hops.
        """
        if arc_from == arc_to:
            return 0.0

        v_from = int(self.arcs[arc_from, 1])
        u_to = int(self.arcs[arc_to, 0])

        if v_from == u_to:
            return 0.0

        # Small local BFS search (up to 3 hops)
        queue = [(v_from, 0.0, 0)]
        visited = {v_from: 0.0}

        while queue:
            curr_node, curr_dist, hops = queue.pop(0)
            if curr_node == u_to:
                return curr_dist

            if hops >= max_hops:
                continue

            for next_node, _, edge_len in self.adj.get(curr_node, []):
                new_dist = curr_dist + edge_len
                if next_node not in visited or new_dist < visited[next_node]:
                    visited[next_node] = new_dist
                    queue.append((next_node, new_dist, hops + 1))

        return float("inf")
