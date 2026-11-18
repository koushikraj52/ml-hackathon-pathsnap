"""
Data loading and spatial coordinate utilities for PathSnap.
Handles .track, .nodes, .arcs, and .route files.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataLoader:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def get_trip_path(self, trip_id: str) -> str:
        return os.path.join(self.raw_data_dir, trip_id)

    def load_trip(self, trip_id: str) -> Dict:
        """
        Loads all 4 raw files for a given trip:
        - track: (N, 3) -> [lon, lat, timestamp]
        - nodes: (M, 2) -> [lon, lat]
        - arcs: (K, 2) -> [from_node, to_node]
        - route: (R,) -> list of traversed arc IDs (if available)
        """
        tdir = self.get_trip_path(trip_id)
        if not os.path.exists(tdir):
            raise FileNotFoundError(f"Trip folder not found: {tdir}")

        track_path = os.path.join(tdir, f"{trip_id}.track")
        nodes_path = os.path.join(tdir, f"{trip_id}.nodes")
        arcs_path = os.path.join(tdir, f"{trip_id}.arcs")
        route_path = os.path.join(tdir, f"{trip_id}.route")

        track = np.loadtxt(track_path, dtype=float)
        if track.ndim == 1:
            track = track.reshape(1, -1)

        nodes = np.loadtxt(nodes_path, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)

        arcs = np.loadtxt(arcs_path, dtype=int)
        if arcs.ndim == 1:
            arcs = arcs.reshape(1, -1)

        route = []
        if os.path.exists(route_path):
            with open(route_path, "r") as f:
                route = [int(line.strip()) for line in f if line.strip()]

        # Coordinate projection scaling factors based on mean latitude
        mean_lat = float(np.mean(nodes[:, 1]))
        kx = 111320.0 * np.cos(np.radians(mean_lat))
        ky = 110540.0

        return {
            "trip_id": trip_id,
            "track": track,
            "nodes": nodes,
            "arcs": arcs,
            "route": route,
            "mean_lat": mean_lat,
            "kx": kx,
            "ky": ky
        }

    def list_available_trips(self) -> List[str]:
        if not os.path.exists(self.raw_data_dir):
            return []
        trips = [
            d for d in os.listdir(self.raw_data_dir)
            if os.path.isdir(os.path.join(self.raw_data_dir, d))
        ]
        return sorted(trips)


def project_point_to_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    kx: float, ky: float
) -> Tuple[float, float, float]:
    """
    Project point P(px, py) onto line segment AB(ax, ay -> bx, by) in local metric space.
    Returns:
      t_clamped: fraction along segment [0.0, 1.0]
      dist: perpendicular / Euclidean distance in meters
      seg_length: physical length of segment AB in meters
    """
    p_x = (px - ax) * kx
    p_y = (py - ay) * ky
    b_x = (bx - ax) * kx
    b_y = (by - ay) * ky

    seg_len_sq = b_x * b_x + b_y * b_y
    seg_length = float(np.sqrt(seg_len_sq))

    if seg_len_sq <= 1e-8:
        dist = float(np.sqrt(p_x * p_x + p_y * p_y))
        return 0.0, dist, 0.0

    t = (p_x * b_x + p_y * b_y) / seg_len_sq
    t_clamped = max(0.0, min(1.0, float(t)))

    proj_x = t_clamped * b_x
    proj_y = t_clamped * b_y
    dist = float(np.sqrt((p_x - proj_x) ** 2 + (p_y - proj_y) ** 2))

    return t_clamped, dist, seg_length


def compute_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Compute initial compass bearing between two points in degrees [0, 360).
    """
    d_lon = np.radians(lon2 - lon1)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)

    y = np.sin(d_lon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lon)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return float(bearing)


def heading_difference(bearing1: float, bearing2: float) -> float:
    """
    Minimum angular difference between two bearings in degrees [0, 180].
    """
    diff = abs(bearing1 - bearing2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    return float(diff)
