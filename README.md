# PathSnap: Distributed Map-Matching & Route Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **PES University — Department of Computer Science & Engineering, RR Campus**  
> **Course:** UE24CS352A: Machine Learning  
> **Challenge:** PathSnap Distributed Map-Matching & Sequential Route Reconstruction  

---

## 📌 Executive Summary

GPS tracking systems in real-world urban and suburban environments regularly suffer from multipath degradation, building occlusion, signal noise, and low temporal sampling rates. Consequently, observed vehicle coordinates frequently deviate onto parallel roads, cut through urban buildings, or jump between disconnected road corridors.

**PathSnap** is a distributed machine learning and Hidden Markov Model (HMM) system that reconstructs coherent, topologically valid vehicle routes by matching noisy GPS trajectory streams to a road network graph. 

### Key System Achievements
- **Candidate Classification Accuracy:** Trained and evaluated **5 distinct ML model families** using **5-Fold GroupKFold Cross-Validation** across 80 trajectory trips (208,417 observations; 2.1M candidate samples).
- **Bayesian Optimization:** Automated hyperparameter tuning with **Optuna (TPE)** over 25 trials, optimizing Group CV Macro F1.
- **HMM Viterbi Sequential Decoding:** Boosted mean observation-level top-1 accuracy from **81.83%** (classifier-only) to **92.01%** (+10.18%), and increased route continuity from **87.77%** to **97.75%** (+9.98%).
- **Two-Laptop Architecture:** Deployed an asynchronous, production-grade **FastAPI backend** (Laptop 1) serving evaluation requests with sub-millisecond per-observation latency ($0.68\text{ ms/obs}$), paired with a **modern web UI** (Laptop 2) featuring interactive Folium maps.
- **Comprehensive Deliverables:** Full submission source code, evaluation artifacts, Jupyter Notebooks, interactive HTML maps, and the consolidated **8-page Technical Report PDF**.

---

## 🏛️ System Architecture

```
                       +---------------------------------------+
                       |             Raw GPS Track             |
                       |       (lon, lat, timestamp, 1Hz)      |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |          Spatial KDTree Index         |
                       |       (cKDTree on Road Network)       |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |      Feature Engineering (10 Feats)   |
                       |   - Orthogonal distance (d_perp)      |
                       |   - Angular difference & cos(d_theta) |
                       |   - Projection ratio along arc (t)    |
                       |   - Segment length & endpoint dists   |
                       |   - Speed & Gaussian emission density |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Tuned HistGradientBoosting Model    |
                       |  Bayesian Optimized (Optuna TPE)      |
                       +-------------------+-------------------+
                                           |  Candidate Probabilities P(z_t | s_t)
                                           v
                       +---------------------------------------+
                       |     Hidden Markov Model & Viterbi     |
                       |   - Hidden States: Candidate Arcs     |
                       |   - Emissions: Model Posterior Probs  |
                       |   - Transitions: Network Topology &   |
                       |     Shortest Path vs Euclidean Metric |
                       +-------------------+-------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |    Globally Optimal Decoded Route     |
                       |  (92.01% Accuracy, 97.75% Continuity) |
                       +-------------------+-------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +---------------------------------------+
|          FastAPI REST Backend         |     |          Modern Evaluator UI          |
|    Hosts Model, Graph & Reference     | <-> |    Selects from 10 Reserved Trips     |
|              [Laptop 1]               |     |       Folium Interactive Map          |
|                                       |     |              [Laptop 2]               |
+---------------------------------------+     +---------------------------------------+
```

---

## 📊 Experimental Results

### 1. Overall Model Comparison (Across 5 Distinct Families)

All models were evaluated on the held-out **Student-Testing Partition (10 unseen trips, 174,138 candidate instances)** following 5-Fold GroupKFold Cross-Validation on the 70 training trips.

| Model | Algorithm Family | Key Configuration | CV Macro F1 | Student Top-1 Acc | Route Continuity | Student Macro F1 | Log Loss | Inference Latency | Training Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | $C=1.0$, balanced, L2 | 0.6287 | 72.69% | 82.16% | 0.7114 | 0.3940 | 0.00005 ms | 0.056 s |
| **Decision Tree** | Tree-based | depth=12, split=20, leaf=10 | 0.7893 | 80.12% | 84.91% | 0.8293 | 0.3655 | 0.00008 ms | 0.293 s |
| **Random Forest** | Ensemble | $n=80$, depth=14, leaf=5 | 0.8328 | 81.34% | 87.49% | 0.8594 | 0.1551 | 0.00077 ms | 0.591 s |
| **HistGradientBoosting** | Boosting (Selected) | lr=0.1, iter=100, depth=8 | **0.8218** | **81.65%** | **87.34%** | **0.8478** | **0.1697** | **0.00069 ms** | **1.434 s** |
| **MLP Neural Net** | Neural Networks | (64, 32), ReLU, Adam | 0.8501 | 81.51% | 87.43% | 0.8828 | 0.1100 | 0.00030 ms | 5.044 s |

---

### 2. Bayesian Hyperparameter Optimization (Optuna TPE)

Bayesian search was conducted on HistGradientBoosting over 25 trials using GroupKFold CV on the training partition:

| Hyperparameter | Search Space | Optimal Value |
| :--- | :---: | :---: |
| `learning_rate` | $[0.03, 0.25]$ (log) | **0.1952** |
| `max_iter` | $[50, 120]$ | **82** |
| `max_depth` | $[6, 14]$ | **8** |
| `min_samples_leaf` | $[15, 60]$ | **27** |
| `l2_regularization` | $[0.01, 5.0]$ (log) | **0.0242** |
| `max_leaf_nodes` | $[20, 64]$ | **52** |

#### Untuned Baseline vs. Tuned Model Comparison

| Configuration | CV Macro F1 | Student Top-1 Acc | Student Macro F1 | Student Route Continuity | Log Loss | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned Baseline** | 0.8218 | 81.65% | 0.8478 | 87.34% | 0.1697 | 0.00067 ms |
| **Tuned (Bayesian Opt)** | **0.8285** | **81.83%** | **0.8385** | **87.77%** | **0.1784** | **0.00073 ms** |

---

### 3. HMM Viterbi Sequential Route Reconstruction Benchmark

Sequential HMM decoding was evaluated across all 10 student testing trips against ground-truth routes:

| Trip ID | Observations | Classifier Top-1 Acc | HMM Viterbi Acc | Accuracy Gain | Classifier Continuity | HMM Continuity | Continuity Gain |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00000011** | 2,828 | 96.11% | **97.98%** | +1.87% | 97.81% | **99.12%** | +1.31% |
| **00000029** | 733 | 82.95% | **93.72%** | +10.78% | 79.78% | **95.77%** | +15.98% |
| **00000037** | 677 | 89.22% | **98.97%** | +9.75% | 89.94% | **98.08%** | +8.14% |
| **00000047** | 1,272 | 78.38% | **89.39%** | +11.01% | 93.71% | **99.61%** | +5.90% |
| **00000057** | 1,645 | 84.92% | **96.23%** | +11.31% | 91.00% | **98.78%** | +7.78% |
| **00000061** | 1,430 | 54.69% | **72.94%** | +18.25% | 73.13% | **91.95%** | +18.82% |
| **00000062** | 3,378 | 83.93% | **96.51%** | +12.58% | 90.55% | **99.17%** | +8.62% |
| **00000069** | 2,067 | 77.60% | **83.45%** | +5.85% | 87.42% | **97.19%** | +9.78% |
| **00000071** | 850 | 90.35% | **96.00%** | +5.65% | 90.69% | **98.23%** | +7.54% |
| **00000080** | 2,510 | 80.16% | **94.90%** | +14.74% | 83.66% | **99.60%** | +15.94% |
| **AVERAGE** | **1,739** | **81.83%** | **92.01%** | **+10.18%** | **87.77%** | **97.75%** | **+9.98%** |

---

## 📁 Repository Structure

```
ml-hackathon-pathsnap/
├── README.md                      # Comprehensive documentation & evaluation instructions
├── requirements.txt               # Pinned Python dependencies
├── data/
│   ├── raw/                       # 80 trip folders (each with .track, .nodes, .arcs, .route)
│   ├── splits/                    # Splits definition (train: 70, student_test: 10, evaluator: 10)
│   └── processed/                 # Cached feature datasets (.joblib)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Parses .track, .nodes, .arcs, .route; local metric projections
│   ├── preprocessor.py            # Missing-value audits, sampling checks, road-network analysis
│   ├── feature_engineering.py    # Spatial KDTree index & 10 engineered geometric/kinematic features
│   ├── models/
│   │   └── __init__.py            # Implementations of all 5 distinct model families
│   ├── training/
│   │   ├── train_all_models.py    # GroupKFold cross-validation & training for all 5 models
│   │   └── bayesian_opt.py        # Optuna TPE Bayesian hyperparameter optimization
│   ├── hmm/
│   │   ├── graph.py               # Road network graph representation & shortest path distance
│   │   └── viterbi.py             # Log-space HMM Viterbi sequential decoder
│   ├── evaluation/
│   │   ├── metrics.py             # Top-1 accuracy, route continuity, Macro F1, Log Loss, miss rate
│   │   └── benchmark.py           # Benchmark script: Classifier vs Classifier+HMM
│   ├── visualization/
│   │   ├── plots.py               # Generates EDA, model comparison, loss curves & HMM gain figures
│   │   └── folium_maps.py         # Multi-layer interactive Folium maps with layer controls & legends
│   ├── api/
│   │   └── app.py                 # FastAPI backend for Laptop 1
│   └── ui/
│       ├── index.html             # Evaluator interface for Laptop 2
│       └── app.js                 # UI client communicating with Laptop 1 API
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_experimentation.ipynb
│   ├── 03_bayesian_optimization.ipynb
│   └── 04_hmm_viterbi_evaluation.ipynb
├── artifacts/
│   ├── models/                    # Serialized trained model weights (.joblib)
│   ├── results/                   # Experimental CSV summaries & Optuna trial logs
│   ├── figures/                   # High-resolution charts & diagnostic plots
│   └── maps/                      # Exported interactive Folium HTML maps
├── report/
│   ├── generate_report.py         # Automated ReportLab PDF generator
│   └── PathSnap_Technical_Report.pdf # Final 8-Page Submission Technical Report
└── scripts/
    ├── run_eda.py                 # Exploratory data analysis runner
    ├── run_training.py            # Model training & comparison runner
    ├── run_bayesian_opt.py        # Bayesian optimization runner
    ├── run_evaluation.py          # HMM Viterbi benchmark runner
    ├── run_api.py                 # FastAPI server launcher
    ├── test_api.py                # Direct API verification script
    └── run_all.py                 # End-to-end master automation runner
```

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/koushikraj52/ml-hackathon-pathsnap.git
cd ml-hackathon-pathsnap

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
To execute the complete pipeline (EDA, Training, Bayesian Optimization, HMM Benchmark, and Report Generation) in one command:
```bash
python scripts/run_all.py
```

---

## 💻 Two-Laptop Deployment Guide

The challenge requires a distributed deployment across two distinct laptops:

### Laptop 1: Model & API Server
Hosts the trained model, road network graph, HMM Viterbi decoder, and server-side ground truth verification.

1. Ensure Laptop 1 and Laptop 2 are on the same Wi-Fi / LAN network.
2. Find Laptop 1's local IP address:
   ```bash
   ipconfig   # On Windows (look for IPv4 Address, e.g., 192.168.1.50)
   ifconfig   # On Linux/macOS
   ```
3. Start the FastAPI backend:
   ```bash
   python scripts/run_api.py --host 0.0.0.0 --port 8000
   ```
4. Verify the API is running by visiting `http://localhost:8000/docs` in your browser.

### Laptop 2: Evaluator User Interface
Hosts the evaluator interface, displays the 10 reserved evaluation trips, and sends inference requests to Laptop 1.

1. Open `src/ui/index.html` in any modern web browser (Chrome, Edge, Firefox).
2. In the top navigation bar, set the **API Server** input to Laptop 1's IP address:
   ```text
   http://192.168.1.50:8000
   ```
3. Click **Ping** to establish a connection. The status indicator will turn green (`Ready`).
4. Select any of the **10 Reserved Evaluation Trips** from the dropdown menu.
5. Click **Run Full Map-Matching Pipeline**.
6. View live KPIs (Top-1 Accuracy, Route Continuity, Latency) and explore the embedded interactive Folium map comparing:
   - **Blue Points:** Original noisy GPS observations (with timestamps and coordinates).
   - **Green Polyline:** Ground-truth reference route.
   - **Dashed Royal Blue Polyline:** PathSnap HMM reconstructed route.
   - **Gray Geometry:** Surrounding road network corridor.

---

## 📡 API Documentation

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/api/health` | Health check, status, loaded model architecture, and available trips count. |
| `GET` | `/api/trips` | Returns list of the 10 reserved evaluation trips with point and arc counts. |
| `POST` | `/api/predict/{trip_id}` | Runs complete pipeline, performs server-side evaluation, and returns metrics. |
| `GET` | `/api/map/{trip_id}` | Serves the interactive Folium HTML map for the selected trip. |

### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/api/predict/00000011"
```

### Example Response:
```json
{
  "trip_id": "00000011",
  "observations_count": 2828,
  "model_used": "best_model_tuned.joblib",
  "metrics": {
    "has_ground_truth": true,
    "classifier_top1_accuracy": 93.46,
    "hmm_viterbi_accuracy": 93.92,
    "accuracy_gain_pct": 0.46,
    "classifier_continuity": 96.5,
    "hmm_viterbi_continuity": 98.02,
    "continuity_gain_pct": 1.52,
    "candidate_miss_rate": 2.69
  },
  "latency": {
    "total_latency_ms": 1929.27,
    "ms_per_observation": 0.682
  },
  "map_url": "/api/map/00000011"
}
```

---

## 📄 Technical Report

The official **8-Page Submission Technical Report PDF** is available at:
👉 **[`report/PathSnap_Technical_Report.pdf`](report/PathSnap_Technical_Report.pdf)**

It comprehensively covers:
1. **Section 2.2:** Dataset Understanding & Exploratory Data Analysis (Preprocessing, Feature Justifications, Missing Values, Folium Maps).
2. **Section 3:** Model Experimentation & Comparison (Consolidated Table 3.1, Compulsory A/B/C Questions for All 5 Models, Figures 2 & 3).
3. **Section 4:** Bayesian Hyperparameter Optimization (Optuna TPE search space, best values, Untuned vs Tuned comparison, Figure 4).
4. **Section 5:** HMM and Viterbi Route Reconstruction (Formulation, Transitions, GPS dropouts, Benchmarks, Figure 5).
5. **Section 6:** Geographical Visualizations with Folium (Layer controls, Legends, Figure 6).
6. **Section 7:** API, UI, and Distributed Deployment (Terminal execution logs, Figure 7).
7. **Section 8:** Academic Integrity & Reproducibility Statement.

---

## 👥 Contributors & Academic Integrity

Developed by the PathSnap Hackathon Team for **UE24CS352A: Machine Learning** at **PES University RR Campus**.  
All implementations, feature transformations, optimization loops, and sequential decoding algorithms were independently engineered.
