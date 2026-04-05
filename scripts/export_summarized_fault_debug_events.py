import os
import csv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Import Shared Configuration and Primitives
#
from fusion_fault_inception_detection.robust_fault_inception_detector import (
    PROJECT_ROOT,
    PROCESSED_DIR,
    FS,
    DATA_S,
    CYCLE_SAMPLES,
    demean_and_detrend,
    window_signal,
    compute_event_features,
    BASELINE_SECONDS,
    MIN_BASELINE_WINDOWS,
    R2_ABS_THRESHOLD,
    R0_ABS_THRESHOLD,
    DELTA_I_THRESHOLD,
    DELTA_U1_SAG_THRESHOLD,
    FFT_Z_THRESHOLD_MIN,
    THD_Z_THRESHOLD_MIN,
    WAVELET_Z_THRESHOLD_MIN,
    DRMS_Z_THRESHOLD_MIN,
    V_STEP_VOLTS,
    I_STEP_AMPS,
    V_IDX,
    I_IDX,
)

# =============================================================================
# 2. Load External Fault Times (Ground Truth / Detector Output)
# =============================================================================
FAULT_TIMES = {}
csv_path = os.path.join(PROCESSED_DIR, "robust_fault_inception_results.csv")

if os.path.exists(csv_path):
    print(f"[INFO] Loading fault times from: {csv_path}")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if row["fault_start_s"]:
                    FAULT_TIMES[int(row["event_idx"])] = float(row["fault_start_s"])
            except ValueError:
                continue
else:
    print(f"[WARN] CSV not found at {csv_path}. Vertical lines will be skipped.")


# =============================================================================
# 3. Define Fault Detection Logic Locally
# =============================================================================
def run_glass_box_detection(features, win_size=CYCLE_SAMPLES, fs=FS):
    """
    Replicates the fusion-score logic from detect_fault_start.py (supervised conditions and weights)
    to expose intermediate voting states for visualization.

    Note: This computes the detector's fusion_score and per-detector contributions,
    excluding the full fault_candidate or persistence gating functionality.
    """
    # Unpack features (must align with compute_event_features outputs)
    I1, I2, I0 = features["I1"], features["I2"], features["I0"]
    Itot, V1 = features["Irms_total"], features["V1"]
    U0 = features.get("U0_rms", None)  # Utilized for resonance blocker, if present
    fft_hf = features["fft_hf_ratio"]
    thd = features["thd"]
    wav = features["wavelet_energy"]

    n_windows = I1.shape[0]
    eps = 1e-12

    # --- Robust baseline window configuration ---
    n_prefault = max(MIN_BASELINE_WINDOWS, int(BASELINE_SECONDS * fs) // win_size)
    n_prefault = min(n_prefault, n_windows)
    baseline_slice = slice(0, n_prefault)

    def robust_mean_std(arr):
        xb = arr[baseline_slice]
        med = float(np.median(xb))
        mad = float(np.median(np.abs(xb - med))) + eps
        std = 1.4826 * mad
        return med, std

    # Compute baselines
    Itot_base_mean, Itot_base_std = robust_mean_std(Itot)
    V1_base_mean, V1_base_std = robust_mean_std(V1)

    # Dead-line and no-voltage guard (replicates detector early abort sequence)
    if V1_base_mean < 1e-2:
        zeros = np.zeros(n_windows, dtype=float)
        contribs = {
            "Unbalance": zeros,
            "Current": zeros,
            "Voltage": zeros,
            "FFT": zeros,
            "THD": zeros,
            "Wavelet": zeros,
            "dRMS": zeros,
        }
        return {"fusion_score": zeros, "contribs": contribs}

    # Compute relative deltas
    Itot_baseline = Itot_base_mean + eps
    V1_baseline = V1_base_mean + eps
    delta_I = Itot / Itot_baseline
    delta_U1 = V1 / V1_baseline

    # I1 safety thresholding (replicates detector handling)
    I1_base_mean, _ = robust_mean_std(I1)
    I1_floor = max(0.10 * abs(I1_base_mean), 1e-2)
    I1_safe = np.where(np.abs(I1) < I1_floor, I1_floor, I1)

    # Compute sequence ratios
    R2 = I2 / (I1_safe + eps)
    R0 = I0 / (I1_safe + eps)

    # Statistical calculations for ratios and deltas
    R2_mean, R2_std = robust_mean_std(R2)
    R0_mean, R0_std = robust_mean_std(R0)
    dI_mean, dI_std = robust_mean_std(delta_I)
    dU1_mean, dU1_std = robust_mean_std(delta_U1)

    # Statistical calculations for FFT and THD
    fft_mean, fft_std = robust_mean_std(fft_hf)
    thd_mean, thd_std = robust_mean_std(thd)

    # Wavelet statistics (replicates detector handling)
    if np.all(np.isnan(wav[baseline_slice])):
        wav_mean, wav_std = 0.0, 1.0
        z_wavelet = np.zeros(n_windows, dtype=float)
    else:
        wav_mean, wav_std = robust_mean_std(wav)
        z_wavelet = (wav - wav_mean) / (wav_std + eps)

    # Calculate dRMS/dt (dItot) and respective z-scores
    dItot = np.zeros_like(Itot)
    dItot[1:] = np.diff(Itot) / (win_size / fs)
    dItot_mean, dItot_std = robust_mean_std(dItot)
    z_dItot = (dItot - dItot_mean) / (dItot_std + eps)

    # Compute Z-scores (replicates detector handling)
    z_R2 = (R2 - R2_mean) / (R2_std + eps)
    z_R0 = (R0 - R0_mean) / (R0_std + eps)
    z_delta_I = (delta_I - dI_mean) / (dI_std + eps)
    z_delta_U1 = (delta_U1 - dU1_mean) / (dU1_std + eps)
    z_fft = (fft_hf - fft_mean) / (fft_std + eps)
    z_thd = (thd - thd_mean) / (thd_std + eps)

    # -------------------------------
    # Supervision and blocker conditions
    # -------------------------------
    # Establish robust current reference
    I_REF = max(I1_base_mean, 0.5)

    # Resonance blocker utilizing U0_ratio, if available
    if U0 is None:
        is_resonance_likely = np.zeros(n_windows, dtype=bool)
    else:
        U0_ratio = U0 / (V1_base_mean + eps)
        is_resonance_likely = (U0_ratio > 0.05) & (I0 < (I_REF * 0.15))

    # Magnitude supervision logic
    MIN_OP_I2_PU = 0.05
    MIN_OP_I0_PU = 0.05
    is_substantial_I2 = I2 > (I_REF * MIN_OP_I2_PU)
    is_substantial_I0 = (I0 > (I_REF * MIN_OP_I0_PU)) & (~is_resonance_likely)

    # Voltage blocker for current condition
    VOLT_HEALTHY_THRESH = 0.92
    is_voltage_healthy = delta_U1 > VOLT_HEALTHY_THRESH

    # -------------------------------
    # Detector conditions
    # -------------------------------
    cond_unbalance = (((R2 > R2_ABS_THRESHOLD) | (z_R2 > 4.0)) & is_substantial_I2) | (
        ((R0 > R0_ABS_THRESHOLD) | (z_R0 > 4.0)) & is_substantial_I0
    )

    cond_voltage = (delta_U1 < DELTA_U1_SAG_THRESHOLD) | (z_delta_U1 < -4.0)

    raw_current_trigger = (delta_I > max(DELTA_I_THRESHOLD, 1.5)) | (z_delta_I > 4.0)
    cond_current = raw_current_trigger & (~is_voltage_healthy)

    cond_fft = (z_fft > max(FFT_Z_THRESHOLD_MIN, 4.0)) & (
        fft_hf > (fft_mean + 3.0 * fft_std)
    )
    cond_thd = (z_thd > max(THD_Z_THRESHOLD_MIN, 4.0)) & (
        thd > (thd_mean + 3.0 * thd_std)
    )

    if np.all(np.isnan(wav)):
        cond_wavelet = np.zeros(n_windows, dtype=bool)
    else:
        cond_wavelet = (z_wavelet > max(WAVELET_Z_THRESHOLD_MIN, 4.0)) & (
            wav > (wav_mean + 3.0 * wav_std)
        )

    cond_drms = np.abs(z_dItot) > max(DRMS_Z_THRESHOLD_MIN, 4.0)

    # -------------------------------
    # Fusion weights assignment
    # -------------------------------
    weights = {
        "Unbalance": 2.0,
        "Current": 2.0,
        "Voltage": 2.0,
        "FFT": 1.0,
        "THD": 1.0,
        "Wavelet": 1.0,
        "dRMS": 1.0,
    }

    contribs = {
        "Unbalance": weights["Unbalance"] * cond_unbalance.astype(float),
        "Current": weights["Current"] * cond_current.astype(float),
        "Voltage": weights["Voltage"] * cond_voltage.astype(float),
        "FFT": weights["FFT"] * cond_fft.astype(float),
        "THD": weights["THD"] * cond_thd.astype(float),
        "Wavelet": weights["Wavelet"] * cond_wavelet.astype(float),
        "dRMS": weights["dRMS"] * cond_drms.astype(float),
    }

    fusion_score = sum(contribs.values())

    # Return fusion score, contributions, and optional diagnostic fields
    return {
        "fusion_score": fusion_score,
        "contribs": contribs,
        # Optional diagnostic fields for extended plotting flexibility
        "z_R2": z_R2,
        "z_R0": z_R0,
        "z_delta_I": z_delta_I,
        "z_delta_U1": z_delta_U1,
        "z_fft": z_fft,
        "z_thd": z_thd,
        "z_wavelet": z_wavelet,
        "z_drms": z_dItot,
        "R2": R2,
        "R0": R0,
        "delta_I": delta_I,
        "delta_U1": delta_U1,
    }


def infer_focus_time(t_win, debug_data, detected_time=None, threshold=5.0):
    """
    Choose the most relevant time for zooming.
    Priority:
    1) externally provided detected_time
    2) first window crossing fusion threshold
    3) first nonzero fusion activity
    4) peak fusion score
    """
    if detected_time is not None:
        return float(detected_time)

    fusion_score = np.asarray(debug_data["fusion_score"])

    idx = np.where(fusion_score >= threshold)[0]
    if idx.size > 0:
        return float(t_win[idx[0]])

    idx = np.where(fusion_score > 0)[0]
    if idx.size > 0:
        return float(t_win[idx[0]])

    return float(t_win[np.argmax(fusion_score)])


def compute_zoom_limits(
    t,
    t_win,
    detrended,
    debug_data,
    detected_time=None,
    pre_s=0.08,
    post_s=0.20,
    min_span_s=0.18,
    current_pad_ratio=0.10,
):
    """
    Compute x/y limits for a zoomed-in IEEE figure.
    """
    focus_t = infer_focus_time(
        t_win, debug_data, detected_time=detected_time, threshold=5.0
    )

    x0 = max(0.0, focus_t - pre_s)
    x1 = min(float(t[-1]), focus_t + post_s)

    # Enforce a minimum zoom width
    if (x1 - x0) < min_span_s:
        half = min_span_s / 2.0
        x0 = max(0.0, focus_t - half)
        x1 = min(float(t[-1]), focus_t + half)

    # If clipping at record edges shrinks the range too much, rebalance
    if (x1 - x0) < min_span_s:
        if x0 <= 0.0:
            x1 = min(float(t[-1]), min_span_s)
        elif x1 >= float(t[-1]):
            x0 = max(0.0, float(t[-1]) - min_span_s)

    # Current-axis limits from zoomed region only
    mask_t = (t >= x0) & (t <= x1)
    zoom_curr = detrended[3:6, mask_t]

    if zoom_curr.size > 0:
        y_min = float(np.min(zoom_curr))
        y_max = float(np.max(zoom_curr))
    else:
        y_min, y_max = -1.0, 1.0

    span = y_max - y_min
    if span < 1e-9:
        span = max(abs(y_min), abs(y_max), 1.0)

    pad = current_pad_ratio * span
    y_curr = [y_min - pad, y_max + pad]

    # Fusion-axis limits from zoomed region only
    mask_w = (t_win >= x0) & (t_win <= x1)
    zoom_score = np.asarray(debug_data["fusion_score"])[mask_w]

    if zoom_score.size > 0:
        y2_max = max(6.0, float(np.max(zoom_score)) + 1.0)
    else:
        y2_max = 6.0

    y_score = [0.0, y2_max]

    return {
        "focus_t": focus_t,
        "x_range": [x0, x1],
        "y_curr": y_curr,
        "y_score": y_score,
    }


# =============================================================================
# 4. IEEE Standard Plot Generation
# =============================================================================
def generate_ieee_paper_plots(
    evt_idx, t, t_win, detrended, debug_data, detected_time, out_folder
):
    """
    Generates a two-panel IEEE-standard figure zoomed around the relevant area.
    """

    font_dict = dict(family="Times New Roman", size=18, color="black")

    zoom = compute_zoom_limits(
        t=t,
        t_win=t_win,
        detrended=detrended,
        debug_data=debug_data,
        detected_time=detected_time,
        pre_s=0.08,  # context before onset
        post_s=0.20,  # context after onset
        min_span_s=0.18,  # avoid over-tight crops
        current_pad_ratio=0.10,
    )

    x_range = zoom["x_range"]
    y_curr = zoom["y_curr"]
    y_score = zoom["y_score"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "<b>(a) Phase Current Dynamics</b>",
            "<b>(b) Fusion Logic Score</b>",
        ),
    )

    # --- Panel 1: Phase Currents ---
    colors_ph = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    phases = ["a", "b", "c"]

    for i, phase in enumerate(phases):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=detrended[i + 3],
                name=f"<i>i<sub>{phase}</sub></i>",
                line=dict(width=1.4, color=colors_ph[i]),
                legendgroup="currents",
            ),
            row=1,
            col=1,
        )

    # --- Panel 2: Fusion Logic Score ---
    fusion_colors = {
        "Unbalance": "#8A2BE2",
        "Current": "#00CED1",
        "Voltage": "#FF69B4",
        "FFT": "#D2691E",
        "THD": "#2F4F4F",
        "Wavelet": "#A9A9A9",
        "dRMS": "#000080",
    }

    for name, val in debug_data["contribs"].items():
        fig.add_trace(
            go.Bar(
                x=t_win,
                y=val,
                name=name,
                marker_color=fusion_colors.get(name, "gray"),
                legendgroup="logic",
            ),
            row=2,
            col=1,
        )

    # Threshold line
    fig.add_hline(
        y=5.0,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="Threshold",
        annotation_position="top right",
    )

    # Detected onset line
    if detected_time is not None:
        for r in [1, 2]:
            fig.add_vline(
                x=detected_time,
                line_width=3,
                line_dash="dash",
                line_color="red",
                opacity=1.0,
            )

    fig.update_layout(
        height=700,
        width=900,
        template="simple_white",
        font=font_dict,
        barmode="stack",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            traceorder="normal",
            font=dict(size=14),
        ),
        margin=dict(l=110, r=20, t=50, b=100),
    )

    # Y axes
    fig.update_yaxes(
        title_text="",
        row=1,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="black",
        range=y_curr,
    )

    fig.update_yaxes(
        title_text="",
        row=2,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=y_score,
    )

    # Manually aligned y-labels
    fig.add_annotation(
        x=-0.1,
        y=0.5,
        xref="paper",
        yref="y domain",
        text="Current (A)",
        textangle=-90,
        showarrow=False,
        font=font_dict,
    )

    fig.add_annotation(
        x=-0.1,
        y=0.5,
        xref="paper",
        yref="y2 domain",
        text="Anomaly Score",
        textangle=-90,
        showarrow=False,
        font=font_dict,
    )

    # X axes: zoom in here
    fig.update_xaxes(
        title_text="Time (s)",
        row=2,
        col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=x_range,
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        range=x_range,
        row=1,
        col=1,
    )

    filename = f"ieee_fig_event_{evt_idx:03d}.pdf"
    full_path = os.path.join(out_folder, filename)

    fig.write_image(full_path)
    fig.write_image(full_path.replace("paper", "editable").replace(".pdf", ".svg"))

    print(f"   [SAVED] {full_path}")


# =============================================================================
# 5. Main Execution Loop
# =============================================================================

IEEE_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "figures",
    "paper",
)
os.makedirs(IEEE_OUTPUT_DIR, exist_ok=True)

IDS_TO_PLOT = [0, 1, 2, 3, 4, 5]
print(f"Generating IEEE plots for events: {IDS_TO_PLOT}")

for evt_idx in IDS_TO_PLOT:
    if evt_idx >= DATA_S.shape[0]:
        continue
    print(f"Processing Event {evt_idx}...")

    # Step 1: Signal Processing
    raw_event = DATA_S[evt_idx].astype(float)

    # Convert ADC counts to physical units
    raw_event[V_IDX, :] *= V_STEP_VOLTS
    raw_event[I_IDX, :] *= I_STEP_AMPS
    detrended = np.zeros_like(raw_event)
    for ch in range(6):
        detrended[ch] = demean_and_detrend(raw_event[ch])

    windows = window_signal(detrended)
    feats = compute_event_features(windows)

    # Step 2: Logic and Time Array Generation
    debug_data = run_glass_box_detection(feats)
    detected_time = FAULT_TIMES.get(evt_idx, None)

    t = np.arange(raw_event.shape[1]) / FS
    n_wins = windows.shape[0]
    t_win = (np.arange(n_wins) + 0.5) * CYCLE_SAMPLES / FS

    # Step 3: Plot Generation
    generate_ieee_paper_plots(
        evt_idx, t, t_win, detrended, debug_data, detected_time, IEEE_OUTPUT_DIR
    )

print("\n[SUCCESS] All plots saved.")
