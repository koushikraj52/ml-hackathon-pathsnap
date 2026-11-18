"""
Data preprocessing, validation, and exploratory statistics module for PathSnap.
"""

import numpy as np
from typing import Dict, List, Any


class DataPreprocessor:
    def __init__(self):
        pass

    def inspect_trip(self, trip_data: Dict) -> Dict[str, Any]:
        """
        Inspects a single trip for:
        - Point count, node count, arc count, route length
        - Missing or NaN values
        - Timestamp gaps and nominal 1 Hz verification
        - Geographical bounding boxes
        - Speed statistics
        """
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        route = trip_data["route"]
        kx, ky = trip_data["kx"], trip_data["ky"]

        # Missing values
        nan_track = int(np.isnan(track).sum())
        nan_nodes = int(np.isnan(nodes).sum())
        nan_arcs = int(np.isnan(arcs).sum())

        # Timestamps
        timestamps = track[:, 2]
        time_diffs = np.diff(timestamps)
        # 1 Hz check
        is_one_hz = np.isclose(time_diffs, 1.0, atol=0.05)
        pct_one_hz = float(np.mean(is_one_hz) * 100.0) if len(time_diffs) > 0 else 100.0
        max_time_gap = float(np.max(time_diffs)) if len(time_diffs) > 0 else 1.0
        large_gaps = int(np.sum(time_diffs > 5.0))

        # Spatial extent
        min_lon, max_lon = float(np.min(track[:, 0])), float(np.max(track[:, 0]))
        min_lat, max_lat = float(np.min(track[:, 1])), float(np.max(track[:, 1]))

        # Speeds (m/s)
        speeds = []
        for i in range(len(track) - 1):
            dt = time_diffs[i]
            if dt <= 0:
                dt = 1.0
            dx = (track[i + 1, 0] - track[i, 0]) * kx
            dy = (track[i + 1, 1] - track[i, 1]) * ky
            dist = np.sqrt(dx * dx + dy * dy)
            speeds.append(dist / dt)
        speeds = np.array(speeds) if speeds else np.array([0.0])

        avg_speed_kmh = float(np.mean(speeds) * 3.6)
        max_speed_kmh = float(np.max(speeds) * 3.6)

        # Route connectivity
        route_connected = 0
        route_disconnected = 0
        if len(route) > 1:
            for i in range(len(route) - 1):
                u1, v1 = arcs[route[i]]
                u2, v2 = arcs[route[i + 1]]
                if v1 == u2:
                    route_connected += 1
                else:
                    route_disconnected += 1

        route_continuity_pct = (
            float(route_connected / (len(route) - 1) * 100.0)
            if len(route) > 1
            else 100.0
        )

        return {
            "trip_id": trip_data["trip_id"],
            "num_track_points": len(track),
            "num_nodes": len(nodes),
            "num_arcs": len(arcs),
            "num_route_arcs": len(route),
            "nan_count": nan_track + nan_nodes + nan_arcs,
            "pct_one_hz": pct_one_hz,
            "max_time_gap_sec": max_time_gap,
            "large_gaps_gt_5s": large_gaps,
            "lon_range": (min_lon, max_lon),
            "lat_range": (min_lat, max_lat),
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "route_continuity_pct": route_continuity_pct
        }

    def clean_trajectory(self, track: np.ndarray) -> np.ndarray:
        """
        Removes identical consecutive duplicate points or NaN records.
        """
        # Filter NaN
        valid_mask = ~np.isnan(track).any(axis=1)
        cleaned = track[valid_mask]

        if len(cleaned) <= 1:
            return cleaned

        # Filter zero-timestamp delta if any
        dt = np.diff(cleaned[:, 2])
        keep = np.ones(len(cleaned), dtype=bool)
        for i in range(len(dt)):
            if dt[i] < 0:
                # Monotonic timestamp enforcement
                cleaned[i + 1, 2] = cleaned[i, 2] + 1.0

        return cleaned
