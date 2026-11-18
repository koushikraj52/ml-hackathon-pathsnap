"""
Folium interactive geographical map generator for PathSnap.
Visualizes original GPS points, road-network corridor, reference route, and predicted route.
"""

import os
import folium
import numpy as np
from typing import Dict, List, Optional


class FoliumMapGenerator:
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.maps_dir = os.path.join(repo_dir, "artifacts", "maps")
        os.makedirs(self.maps_dir, exist_ok=True)

    def create_trip_map(
        self,
        trip_id: str,
        track: np.ndarray,
        nodes: np.ndarray,
        arcs: np.ndarray,
        reference_route_arcs: Optional[List[int]] = None,
        predicted_route_arcs: Optional[List[int]] = None,
        save_filename: Optional[str] = None
    ) -> folium.Map:
        """
        Creates an interactive multi-layer Folium map for a trip:
        - Layer 1: Road Network Geometry (corridor arcs around trajectory)
        - Layer 2: Raw GPS Track (Blue circle markers with timestamps and coordinates)
        - Layer 3: Ground-Truth Reference Route (Solid Green polyline, 4px)
        - Layer 4: Predicted Reconstructed Route (Solid Royal Blue polyline, 3px)
        - Interactive Legend & LayerControl
        """
        center_lat = float(np.mean(track[:, 1]))
        center_lon = float(np.mean(track[:, 0]))

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap"
        )

        min_lon, max_lon = float(np.min(track[:, 0])), float(np.max(track[:, 0]))
        min_lat, max_lat = float(np.min(track[:, 1])), float(np.max(track[:, 1]))

        # 1. FeatureGroup: Road Network Geometry (arcs related to routes)
        fg_network = folium.FeatureGroup(name="Road Network Geometry", show=False)
        relevant_arcs = set()
        if reference_route_arcs:
            relevant_arcs.update(reference_route_arcs[:200])
        if predicted_route_arcs:
            relevant_arcs.update(predicted_route_arcs[:200])

        for aidx in relevant_arcs:
            if aidx < len(arcs):
                u, v = arcs[aidx]
                fg_network.add_child(folium.PolyLine(
                    locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                    color="#94a3b8",
                    weight=2,
                    opacity=0.6,
                    tooltip=f"Arc {aidx}: {u}->{v}"
                ))
        fg_network.add_to(fmap)

        # 2. FeatureGroup: Raw GPS Track
        fg_gps = folium.FeatureGroup(name="Observed GPS Track (Noisy Points)", show=True)
        gps_coords = [[pt[1], pt[0]] for pt in track]
        fg_gps.add_child(folium.PolyLine(
            locations=gps_coords,
            color="#3b82f6",
            weight=1.5,
            dash_array="4, 4",
            opacity=0.6
        ))

        step = max(1, len(track) // 200)
        for i in range(0, len(track), step):
            pt = track[i]
            fg_gps.add_child(folium.CircleMarker(
                location=[pt[1], pt[0]],
                radius=3,
                color="#1d4ed8",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.8,
                tooltip=f"GPS #{i} | Time: {pt[2]:.1f}s | ({pt[1]:.5f}, {pt[0]:.5f})"
            ))
        fg_gps.add_to(fmap)

        # 3. FeatureGroup: Reference Route (Ground Truth)
        if reference_route_arcs:
            fg_gt = folium.FeatureGroup(name="Ground-Truth Route (Green)", show=True)
            for aidx in reference_route_arcs:
                if aidx < len(arcs):
                    u, v = arcs[aidx]
                    fg_gt.add_child(folium.PolyLine(
                        locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                        color="#16a34a",
                        weight=5,
                        opacity=0.85,
                        tooltip=f"Ground Truth Arc {aidx}"
                    ))
            fg_gt.add_to(fmap)

        # 4. FeatureGroup: Predicted Reconstructed Route
        if predicted_route_arcs:
            fg_pred = folium.FeatureGroup(name="Predicted Route (PathSnap HMM)", show=True)
            unique_pred_arcs = []
            for a in predicted_route_arcs:
                if not unique_pred_arcs or unique_pred_arcs[-1] != a:
                    unique_pred_arcs.append(a)

            for aidx in unique_pred_arcs:
                if aidx < len(arcs):
                    u, v = arcs[aidx]
                    fg_pred.add_child(folium.PolyLine(
                        locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                        color="#2563eb",
                        weight=3.5,
                        opacity=0.9,
                        dash_array="6, 3",
                        tooltip=f"Predicted Arc {aidx}"
                    ))
            fg_pred.add_to(fmap)

        fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        folium.LayerControl(position="topright", collapsed=False).add_to(fmap)

        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            width: 260px;
            background-color: white;
            border: 2px solid #cbd5e1;
            border-radius: 8px;
            z-index: 9999;
            font-size: 13px;
            font-family: Arial, sans-serif;
            padding: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        ">
            <h4 style="margin: 0 0 8px 0; font-size: 14px; font-weight: bold; color: #1e293b;">PathSnap Trip {trip_id}</h4>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="background: #3b82f6; width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
                <span>Observed GPS Points</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="background: #16a34a; width: 22px; height: 4px; display: inline-block; margin-right: 8px;"></span>
                <span>Ground Truth Route</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="border-top: 3px dashed #2563eb; width: 22px; height: 0px; display: inline-block; margin-right: 8px;"></span>
                <span>PathSnap Predicted Route</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="background: #94a3b8; width: 22px; height: 2px; display: inline-block; margin-right: 8px;"></span>
                <span>Road Network Geometry</span>
            </div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

        if save_filename:
            out_path = os.path.join(self.maps_dir, save_filename)
            fmap.save(out_path)
            print(f"Saved interactive Folium map to {out_path}")

        return fmap
