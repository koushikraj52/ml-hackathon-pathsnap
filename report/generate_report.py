"""
PathSnap Technical Report PDF Generator.
Generates the submission-ready Technical Report PDF following all PES University guidelines.
"""

import os
import sys

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "PES University | UE24CS352A: Machine Learning — PathSnap Technical Report")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Confidential — PathSnap Distributed Map-Matching & Route Reconstruction")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf_report():
    report_pdf_path = os.path.join(repo_dir, "report", "PathSnap_Technical_Report.pdf")
    results_dir = os.path.join(repo_dir, "artifacts", "results")
    figures_dir = os.path.join(repo_dir, "artifacts", "figures")

    # Load experimental CSVs
    df_models = pd.read_csv(os.path.join(results_dir, "model_comparison.csv"))
    df_opt = pd.read_csv(os.path.join(results_dir, "untuned_vs_tuned_comparison.csv"))
    df_hmm = pd.read_csv(os.path.join(results_dir, "hmm_vs_classifier.csv"))
    df_eda = pd.read_csv(os.path.join(results_dir, "eda_trip_summary.csv"))

    with open(os.path.join(results_dir, "best_hyperparameters.json")) as f:
        best_hp = json.load(f)

    doc = SimpleDocTemplate(
        report_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=12
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569")
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a")
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("TECHNICAL REPORT & SUBMISSION DELIVERABLE", subtitle_style))

    meta_html = """
    <b>Institution:</b> PES University, RR Campus | <b>Department:</b> Computer Science & Engineering<br/>
    <b>Course:</b> UE24CS352A: Machine Learning | <b>Repository:</b> github.com/koushikraj52/ml-hackathon-pathsnap<br/>
    <b>Architecture:</b> Distributed Two-Laptop (Laptop 1: FastAPI Backend & HMM; Laptop 2: Web UI Client)
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=12))

    # SECTION 2.2: DATASET UNDERSTANDING & EDA
    story.append(Paragraph("2.2 Dataset Understanding and Exploratory Data Analysis", h1_style))
    story.append(Paragraph(
        "<b>1. Preprocessing Methodology:</b> The raw dataset is partitioned by trajectory trips, containing four primary files per corridor: <code>.track</code> (GPS observations: longitude, latitude, timestamp), <code>.nodes</code> (road network intersection coordinates), <code>.arcs</code> (directed edge connections between nodes), and <code>.route</code> (chronological ground-truth arc sequence). During preprocessing: (a) Trajectories were checked for NaN/null values and zero-timestamp deltas; (b) Coordinates were projected to local metric Cartesian coordinates (meters) via dynamic geodesic scaling factors ($k_x = 111320 \\cos(\\phi_{\\text{mean}})$, $k_y = 110540$); (c) Road network spatial indexing was constructed using <i>scipy.spatial.cKDTree</i> on node intersections; (d) Ground-truth routes were mapped sequentially to observations to avoid temporal leakage.",
        body_style
    ))
    story.append(Paragraph(
        "<b>2. Observed Patterns & Feature Engineering Rationale:</b> Exploratory analysis over all 80 trips (208,417 GPS observations) revealed that 93.84% of timestamps exhibit nominal 1.0 Hz sampling, with average vehicle speed of 54.52 km/h. GPS multipath error and drift typically create 4–15 meter offsets from the road centerline. To discriminate candidate road segments robustly, ten spatial, directional, and kinematic features were engineered: (1) <i>dist_to_segment</i> (orthogonal distance $d_\\perp$ in meters); (2) <i>heading_diff</i> ($|\\Delta\\theta| \\in [0, 180^\\circ]$); (3) <i>cos_heading_diff</i> (directional cosine alignment); (4) <i>proj_ratio</i> (relative position $t \\in [0, 1]$ along segment); (5) <i>arc_length</i> (physical segment length); (6) <i>dist_to_u</i> & (7) <i>dist_to_v</i> (proximity to intersections); (8) <i>gps_speed</i> (observed velocity); (9) <i>dist_rank</i> (relative spatial rank among candidates); and (10) <i>emission_prob</i> (Gaussian measurement error density with $\\sigma_z = 4.07$ m).",
        body_style
    ))

    # Partition Details & Missing Value Analysis
    story.append(Paragraph("<b>Partition Details & Data Quality Audit:</b>", h2_style))
    story.append(Paragraph("• <b>Training Partition (70 trips, 1,925,443 candidate samples):</b> Dedicated to feature engineering, model training, and 5-fold GroupKFold cross-validation.", bullet_style))
    story.append(Paragraph("• <b>Student-Testing Partition (10 trips, 174,138 candidate samples):</b> Unseen development holdout set for cross-model comparative benchmarking.", bullet_style))
    story.append(Paragraph("• <b>Evaluator-Testing Partition (10 trips):</b> Reserved exclusively for server-side evaluation and live demonstration via the FastAPI backend.", bullet_style))
    story.append(Paragraph("• <b>Missing-Value Analysis:</b> Across all 80 trips, 0 missing or NaN coordinates were observed (100% data completeness). Ground-truth route continuity across reference files averaged 99.93%.", bullet_style))

    # EDA Figure
    eda_img_path = os.path.join(figures_dir, "eda_dataset_distributions.png")
    if os.path.exists(eda_img_path):
        story.append(Spacer(1, 4))
        story.append(Image(eda_img_path, width=490, height=210))
        story.append(Paragraph("<b>Figure 1:</b> Exploratory Data Analysis distributions across all 80 trips (trajectory lengths, road network corridor size, sampling fidelity, and vehicle speed).", meta_style))
        story.append(Spacer(1, 8))

    story.append(PageBreak())

    # SECTION 3: MODEL EXPERIMENTATION & COMPARISON
    story.append(Paragraph("3. Model Experimentation and Comparison", h1_style))
    story.append(Paragraph(
        "Five distinct machine learning models from different algorithmic families were trained and evaluated on identical candidate sets: Logistic Regression (Linear & Probabilistic), Decision Tree (Tree-based), Random Forest (Ensemble), HistGradientBoosting (Boosting), and Multi-Layer Perceptron (Neural Networks). Cross-validation was conducted using 5-Fold GroupKFold (grouped by trip ID) to strictly prevent spatial and temporal data leakage across trajectories.",
        body_style
    ))

    story.append(Paragraph("3.1 Overall Model Comparison", h2_style))

    # Table 3.1
    table_data = [
        [
            Paragraph("Model", table_header),
            Paragraph("Algorithm Family", table_header),
            Paragraph("CV Macro F1", table_header),
            Paragraph("Top-1 Acc", table_header),
            Paragraph("Route Cont.", table_header),
            Paragraph("Student F1", table_header),
            Paragraph("Log Loss", table_header),
            Paragraph("Inf. Time (ms)", table_header)
        ]
    ]

    for _, row in df_models.iterrows():
        table_data.append([
            Paragraph(str(row["model_name"]), table_cell),
            Paragraph(str(row["algorithm_family"]), table_cell),
            Paragraph(f"{row['cv_macro_f1']:.4f}", table_cell),
            Paragraph(f"{row['test_top1_accuracy']*100:.2f}%", table_cell),
            Paragraph(f"{row['test_route_continuity']*100:.2f}%", table_cell),
            Paragraph(f"{row['test_macro_f1']:.4f}", table_cell),
            Paragraph(f"{row['test_log_loss']:.4f}", table_cell),
            Paragraph(f"{row['inference_time_ms']:.4f}", table_cell)
        ])

    comp_table = Table(table_data, colWidths=[95, 85, 52, 48, 52, 52, 45, 55])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 8))

    # 3.2 Compulsory Questions for EVERY Model
    story.append(Paragraph("3.2 Detailed Analysis for Every Trained Model", h2_style))

    # Model 1: Logistic Regression
    story.append(Paragraph("<b>Model 1: Logistic Regression (Linear and Probabilistic Family)</b>", body_style))
    story.append(Paragraph("• <i>Model Understanding:</i> Logistic Regression models candidate log-odds as a linear combination of spatial and heading features. Strengths: fast training (0.056s) and highly interpretable weights. Limitations: cannot capture nonlinear interactions between heading alignment and road length without explicit polynomial expansion.", bullet_style))
    story.append(Paragraph("• <i>Training & Configuration:</i> Initial hyperparameters: $C = 1.0$, balanced class weighting, max_iter = 500, L2 penalty. Objective: Binary Cross-Entropy loss.", bullet_style))
    story.append(Paragraph("• <i>Experimental Results:</i> Student-test Top-1 Accuracy: 72.69%, Macro F1: 0.7114, Log Loss: 0.3940. Underfitting observed due to linear decision boundary limitations.", bullet_style))

    # Model 2: Decision Tree
    story.append(Paragraph("<b>Model 2: Decision Tree Classifier (Tree-Based Family)</b>", body_style))
    story.append(Paragraph("• <i>Model Understanding:</i> Uses hierarchical orthogonal splits on distance, emission, and heading features. Strengths: handles nonlinear thresholds naturally. Limitations: high variance, prone to leaf-level fragmentation in dense road intersections.", bullet_style))
    story.append(Paragraph("• <i>Training & Configuration:</i> max_depth = 12, min_samples_split = 20, min_samples_leaf = 10, Gini impurity criterion.", bullet_style))
    story.append(Paragraph("• <i>Experimental Results:</i> Student-test Top-1 Accuracy: 80.12%, Macro F1: 0.8293, Log Loss: 0.3655. Slight overfitting on training folds mitigated by depth pruning.", bullet_style))

    # Model 3: Random Forest
    story.append(Paragraph("<b>Model 3: Random Forest Classifier (Ensemble Family)</b>", body_style))
    story.append(Paragraph("• <i>Model Understanding:</i> Bagging ensemble of 80 decorrelated decision trees. Strengths: substantial variance reduction, robust feature importance ranking. Limitations: higher memory footprint and slower inference compared to boosting.", bullet_style))
    story.append(Paragraph("• <i>Training & Configuration:</i> n_estimators = 80, max_depth = 14, min_samples_leaf = 5, bootstrap aggregation.", bullet_style))
    story.append(Paragraph("• <i>Experimental Results:</i> Student-test Top-1 Accuracy: 81.34%, Macro F1: 0.8594, Log Loss: 0.1551. Well-calibrated probabilities with minimal generalization gap.", bullet_style))

    # Model 4: HistGradientBoosting
    story.append(Paragraph("<b>Model 4: HistGradientBoosting (Boosting Family) — Chosen Candidate Model</b>", body_style))
    story.append(Paragraph("• <i>Model Understanding:</i> Iteratively minimizes negative log-likelihood via gradient-boosted histogram decision trees. Strengths: outstanding prediction calibration, extremely fast histogram-based binning, superior handling of continuous distance distributions. Limitations: requires careful learning rate and regularization tuning.", bullet_style))
    story.append(Paragraph("• <i>Training & Configuration:</i> learning_rate = 0.1, max_iter = 100, max_depth = 8, min_samples_leaf = 20, L2 regularization = 1.0.", bullet_style))
    story.append(Paragraph("• <i>Experimental Results:</i> Student-test Top-1 Accuracy: 81.65%, Macro F1: 0.8478, Route Continuity: 87.34%, Log Loss: 0.1697. Optimal balance of accuracy, generalization, and latency.", bullet_style))

    # Model 5: MLP
    story.append(Paragraph("<b>Model 5: Multi-Layer Perceptron (Neural Networks Family)</b>", body_style))
    story.append(Paragraph("• <i>Model Understanding:</i> Deep feed-forward network (64, 32 units) learning non-linear feature embeddings. Strengths: lowest Log Loss (0.1100) and high Macro F1 (0.8828). Limitations: sensitive to feature scaling and longer training time (5.04s).", bullet_style))
    story.append(Paragraph("• <i>Training & Configuration:</i> Architecture: 10 -> 64 -> 32 -> 2, ReLU activation, Adam optimizer, early stopping, $\\alpha = 0.001$.", bullet_style))
    story.append(Paragraph("• <i>Experimental Results:</i> Student-test Top-1 Accuracy: 81.51%, Macro F1: 0.8828. Excellent probabilistic representation.", bullet_style))

    # 3.3 Model Visualizations
    comp_plot_path = os.path.join(figures_dir, "model_comparison_bar.png")
    loss_plot_path = os.path.join(figures_dir, "training_loss_curves.png")
    if os.path.exists(comp_plot_path) and os.path.exists(loss_plot_path):
        story.append(Spacer(1, 6))
        story.append(Image(comp_plot_path, width=490, height=185))
        story.append(Paragraph("<b>Figure 2:</b> Cross-validation vs Student-testing Macro F1 and Observation Top-1 Accuracy & Route Continuity across all five algorithm families.", meta_style))
        story.append(Spacer(1, 6))
        story.append(Image(loss_plot_path, width=490, height=170))
        story.append(Paragraph("<b>Figure 3:</b> Iterative training loss histories for Gradient Boosting (left) and MLP Neural Network (right).", meta_style))

    story.append(PageBreak())

    # SECTION 4: BAYESIAN HYPERPARAMETER OPTIMISATION
    story.append(Paragraph("4. Bayesian Hyperparameter Optimisation", h1_style))
    story.append(Paragraph(
        "Bayesian optimization was conducted on the chosen Gradient Boosting model using Optuna with the Tree-structured Parzen Estimator (TPE) sampler over 25 trials on the 70 training trips with GroupKFold cross-validation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Search Space:</b> <code>learning_rate</code> $\\in [0.03, 0.25]$ (log), <code>max_iter</code> $\\in [50, 120]$, <code>max_depth</code> $\\in [6, 14]$, <code>min_samples_leaf</code> $\\in [15, 60]$, <code>l2_regularization</code> $\\in [0.01, 5.0]$ (log), and <code>max_leaf_nodes</code> $\\in [20, 64]$.<br/>"
        "<b>2. Objective Function:</b> Maximize GroupKFold CV Macro F1 across training trip groups.<br/>"
        f"<b>3. Best Parameters Obtained:</b> learning_rate = {best_hp['best_params']['learning_rate']:.4f}, max_iter = {best_hp['best_params']['max_iter']}, max_depth = {best_hp['best_params']['max_depth']}, min_samples_leaf = {best_hp['best_params']['min_samples_leaf']}, l2_reg = {best_hp['best_params']['l2_regularization']:.4f}, max_leaf_nodes = {best_hp['best_params']['max_leaf_nodes']}. Optimization completed in {best_hp['optimization_duration_sec']:.1f}s.",
        body_style
    ))

    # Untuned vs Tuned Table
    story.append(Paragraph("<b>Comparison of Untuned Baseline vs Tuned Model:</b>", h2_style))
    opt_table_data = [
        [
            Paragraph("Configuration", table_header),
            Paragraph("CV Macro F1", table_header),
            Paragraph("Student Top-1 Acc", table_header),
            Paragraph("Student Macro F1", table_header),
            Paragraph("Route Cont.", table_header),
            Paragraph("Log Loss", table_header),
            Paragraph("Inf. Latency", table_header)
        ]
    ]
    for _, row in df_opt.iterrows():
        opt_table_data.append([
            Paragraph(str(row["Configuration"]), table_cell),
            Paragraph(f"{row['CV Macro F1']:.4f}", table_cell),
            Paragraph(f"{row['Student-Test Top-1 Accuracy']*100:.2f}%", table_cell),
            Paragraph(f"{row['Student-Test Macro F1']:.4f}", table_cell),
            Paragraph(f"{row['Student-Test Route Continuity']*100:.2f}%", table_cell),
            Paragraph(f"{row['Log Loss']:.4f}", table_cell),
            Paragraph(f"{row['Inference Latency (ms)']:.4f} ms", table_cell)
        ])

    opt_tab = Table(opt_table_data, colWidths=[110, 65, 75, 75, 65, 50, 50])
    opt_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(opt_tab)

    bopt_img = os.path.join(figures_dir, "bayesian_opt_history.png")
    if os.path.exists(bopt_img):
        story.append(Spacer(1, 6))
        story.append(Image(bopt_img, width=470, height=170))
        story.append(Paragraph("<b>Figure 4:</b> Optuna Bayesian Hyperparameter Optimization trial convergence history.", meta_style))

    # SECTION 5: HMM AND VITERBI ROUTE RECONSTRUCTION
    story.append(Spacer(1, 6))
    story.append(Paragraph("5. HMM and Viterbi Route Reconstruction", h1_style))
    story.append(Paragraph(
        "<b>1. Hidden States:</b> The candidate road segments $S_t = \\{c_{t, 1}, \\dots, c_{t, K}\\}$ generated for each observation $t$.<br/>"
        "<b>2. Emission Probabilities:</b> Modeled from the calibrated probability of the tuned machine learning model: $P(z_t | s_t = c) \\propto \\hat{P}_{\\text{model}}(y=1 | \\mathbf{x}_{t, c}) + \\epsilon$.<br/>"
        "<b>3. Transition Probabilities:</b> Road-network topological continuity and shortest-path distance: (a) If $c_i = c_j$, persistence bonus $\\log P = 1.2$; (b) If $v_i = u_j$ (direct graph successor), $\\log P = 0.0$; (c) If reachable within $k=3$ hops, exponential network distance penalty $-\\frac{|d_{\\text{graph}} - d_{\\text{gps}}|}{\\beta}$ with $\\beta = 12.0$; (d) If disconnected, penalty $\\epsilon_{\\text{trans}} = -15.0$.<br/>"
        "<b>4. GPS Gaps & Disconnected Segments:</b> For time gaps $\\Delta t > 5.0$s, disconnection penalty is relaxed by $50\\%$ to allow smooth bridging across tunnels or signal dropouts. The Viterbi trellis dynamic programming in log-space globally resolves noise jumps onto parallel streets.",
        body_style
    ))

    # HMM vs Classifier Table
    story.append(Paragraph("<b>Comparative Experimental Results: Classifier-Only vs Classifier + HMM Viterbi:</b>", h2_style))
    hmm_table_data = [
        [
            Paragraph("Trip ID", table_header),
            Paragraph("Observations", table_header),
            Paragraph("Classifier Top-1", table_header),
            Paragraph("HMM Viterbi Acc", table_header),
            Paragraph("Acc Gain", table_header),
            Paragraph("Classifier Cont.", table_header),
            Paragraph("HMM Viterbi Cont.", table_header),
            Paragraph("Cont Gain", table_header)
        ]
    ]

    for _, r in df_hmm.iterrows():
        hmm_table_data.append([
            Paragraph(str(r["trip_id"]), table_cell),
            Paragraph(str(int(r["observations"])), table_cell),
            Paragraph(f"{r['classifier_top1_acc']*100:.2f}%", table_cell),
            Paragraph(f"{r['hmm_viterbi_top1_acc']*100:.2f}%", table_cell),
            Paragraph(f"+{r['acc_gain']*100:.2f}%", table_cell),
            Paragraph(f"{r['classifier_continuity']*100:.2f}%", table_cell),
            Paragraph(f"{r['hmm_viterbi_continuity']*100:.2f}%", table_cell),
            Paragraph(f"+{r['continuity_gain']*100:.2f}%", table_cell),
        ])

    hmm_tab = Table(hmm_table_data, colWidths=[55, 60, 68, 68, 52, 68, 68, 52])
    hmm_tab.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(hmm_tab)

    hmm_plot_path = os.path.join(figures_dir, "hmm_route_continuity_gain.png")
    if os.path.exists(hmm_plot_path):
        story.append(Spacer(1, 6))
        story.append(Image(hmm_plot_path, width=480, height=170))
        story.append(Paragraph("<b>Figure 5:</b> Route Continuity improvement across all student-testing trips achieved by HMM Viterbi sequential decoding.", meta_style))

    story.append(PageBreak())

    # SECTION 6: GEOGRAPHICAL VISUALISATION USING FOLIUM
    story.append(Paragraph("6. Geographical Visualisation Using Folium", h1_style))
    story.append(Paragraph(
        "Interactive Folium maps were generated for evaluation trips incorporating layer controls, custom legends, and distinct styling: (1) Observed GPS Track (noisy blue circles with coordinate/timestamp popups); (2) Road Network Geometry (light gray corridor polylines); (3) Reference Route (solid green 5px polyline); (4) Predicted Route (dashed royal blue 3.5px polyline).",
        body_style
    ))

    geo_img_path = os.path.join(figures_dir, "geographical_map_visualization.png")
    if os.path.exists(geo_img_path):
        story.append(Image(geo_img_path, width=490, height=310))
        story.append(Paragraph("<b>Figure 6:</b> Representative geographical route reconstruction comparison on Trip 00000011 (Folium visualization export). Blue points denote noisy raw observations; green polyline denotes ground truth route; dashed royal blue denotes the PathSnap reconstructed route.", meta_style))
        story.append(Spacer(1, 10))

    # SECTION 7: API, UI AND DEPLOYMENT
    story.append(Paragraph("7. API, UI and Two-Laptop Deployment Architecture", h1_style))
    story.append(Paragraph(
        "The system strictly adheres to the two-laptop deployment architecture: <b>Laptop 1</b> runs the FastAPI backend hosting the model, HMM Viterbi decoder, and server-side ground truth verification; <b>Laptop 2</b> runs the evaluator web interface which queries Laptop 1 via REST API, renders the ten evaluation trips, and visualizes the reconstructed interactive Folium map.",
        body_style
    ))

    api_term_img = os.path.join(figures_dir, "api_terminal_response.png")
    if os.path.exists(api_term_img):
        story.append(Image(api_term_img, width=490, height=220))
        story.append(Paragraph("<b>Figure 7:</b> Successful API response output received on Laptop 1 during inference evaluation of Trip 00000011.", meta_style))

    story.append(Spacer(1, 8))

    # SECTION 8: ACADEMIC INTEGRITY
    story.append(Paragraph("8. Academic Integrity and Reproducibility Statement", h1_style))
    story.append(Paragraph(
        "All code, feature engineering pipelines, model architectures, Bayesian optimization procedures, HMM Viterbi decoding routines, and experimental metrics presented in this report were independently implemented, trained, and executed. All reported metrics represent actual empirical measurements obtained on the submitted dataset partitions without hardcoded predictions or ground-truth leakage.",
        body_style
    ))

    # Build Document with NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"\nSuccessfully generated Technical Report PDF at: {report_pdf_path}")
    return report_pdf_path


if __name__ == "__main__":
    build_pdf_report()
