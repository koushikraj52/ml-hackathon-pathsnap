"""
Exploratory Data Analysis runner script for PathSnap.
Analyzes raw datasets across all 80 trips and generates summary tables and plots.
"""

import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import json
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.visualization.plots import PlotGenerator
from src.visualization.folium_maps import FoliumMapGenerator


def main():
    raw_data_dir = os.path.join(repo_dir, "data", "raw")
    results_dir = os.path.join(repo_dir, "artifacts", "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=== Running Exploratory Data Analysis (EDA) on All 80 Trips ===")
    loader = DataLoader(raw_data_dir)
    preprocessor = DataPreprocessor()
    plot_gen = PlotGenerator(repo_dir)
    map_gen = FoliumMapGenerator(repo_dir)

    trips = loader.list_available_trips()
    print(f"Total trips found in raw data: {len(trips)}")

    records = []
    for tid in trips:
        t_data = loader.load_trip(tid)
        stats = preprocessor.inspect_trip(t_data)
        records.append(stats)

    df_eda = pd.DataFrame(records)
    csv_out = os.path.join(results_dir, "eda_trip_summary.csv")
    df_eda.to_csv(csv_out, index=False)
    print(f"Saved EDA trip summary table to {csv_out}")

    # Generate EDA figures
    plot_gen.plot_eda(df_eda)

    # Generate representative EDA Folium map for Trip 00000001
    sample_trip = loader.load_trip("00000001")
    map_gen.create_trip_map(
        trip_id="00000001",
        track=sample_trip["track"],
        nodes=sample_trip["nodes"],
        arcs=sample_trip["arcs"],
        reference_route_arcs=sample_trip["route"],
        save_filename="trip_00000001_eda.html"
    )

    # Summary statistics printout
    print("\nEDA Dataset Inventory & Summary:")
    print(f"  Total Trips: {len(df_eda)}")
    print(f"  Total GPS Observations: {df_eda['num_track_points'].sum():,}")
    print(f"  Avg GPS Observations/Trip: {df_eda['num_track_points'].mean():.1f} (min: {df_eda['num_track_points'].min()}, max: {df_eda['num_track_points'].max()})")
    print(f"  Avg Nodes/Corridor: {df_eda['num_nodes'].mean():.1f}")
    print(f"  Avg Arcs/Corridor: {df_eda['num_arcs'].mean():.1f}")
    print(f"  Nominal 1 Hz Percentage: {df_eda['pct_one_hz'].mean():.2f}%")
    print(f"  Total Missing / NaN values: {df_eda['nan_count'].sum()}")
    print(f"  Avg Vehicle Speed: {df_eda['avg_speed_kmh'].mean():.2f} km/h")
    print(f"  Route Continuity of Ground Truth: {df_eda['route_continuity_pct'].mean():.2f}%\n")


if __name__ == "__main__":
    main()
