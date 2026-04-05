"""
Overview
--------
This module provides the reference implementation for the physics-guided
fusion framework to detect precise fault inception timestamps (t_0) in
unlabeled High-Voltage Digital Fault Recorder (DFR) waveforms.

The system employs a "White-Box" Sensor Fusion architecture integrating:
1. Symmetrical Component Analysis: Quantifying electrical unbalance (ANSI 46/50G).
2. Robust Statistical Normalization: Utilizing Median and MAD to prevent
   statistical masking during severe high-energy disturbances.
3. Time-Frequency Analysis: FFT and Discrete Wavelet Transform (DWT).

The primary objective is unsupervised differentiation between genuine fault
events (Line-to-Ground, Line-to-Line) and operational transients such as
load switching, transformer energization, and measurement noise.
"""

import os
import csv
from typing import Dict, Tuple, Optional, Any

import numpy as np

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================
# PyWavelets is used for multi-resolution analysis of transient energy.
# The system degrades gracefully (disabling wavelet features) if missing.
try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print(
        "[INFO] PyWavelets (pywt) not found. Wavelet-based features will be disabled."
    )

# =============================================================================
# GLOBAL CONFIGURATION & PHYSICAL CONSTANTS
# =============================================================================

# File System Configuration
# NOTE: Update BASE_DIR to point to your local dataset directory before execution.
# IMPORTANT: The detection thresholds, physical constants, and quantization steps
# below are specifically tuned for the Réseau de Transport d'Électricité (RTE) dataset
# (90 kV transmission, 6400 Hz sampling rate, 16-bit encoding).
# Dataset Source: https://github.com/rte-france/digital-fault-recording-database

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DATA_S_PATH = os.path.join(RAW_DATA_DIR, "DATA_S.npz")

# -----------------------------------------------------------------------------
# Signal Processing Parameters
# -----------------------------------------------------------------------------
FS = 6400.0  # Sampling frequency [Hz]
F_NOM = 50.0  # Nominal grid frequency [Hz]

# Cycle synchronization: 6400 Hz / 50 Hz = 128 samples per cycle.
# Precise alignment minimizes spectral leakage during Phasor/FFT computation.
CYCLE_SAMPLES = int(FS / F_NOM)
CYCLE_DT = CYCLE_SAMPLES / FS  # Fundamental period [s] (20ms for 50Hz)

MA_WINDOW = 256  # Window length for moving-average detrending

# -----------------------------------------------------------------------------
# Persistence and Validation Gating
# -----------------------------------------------------------------------------
PERSISTENCE_WINDOWS = 4  # Minimum duration to confirm a sustained fault (80 ms)
R2_PERSIST_THRESHOLD = 0.15  # Min Negative Sequence ratio (I2/I1) for persistence
R0_PERSIST_THRESHOLD = 0.05  # Min Zero Sequence ratio (I0/I1) for persistence
U0_PERSIST_Z = 3.0  # Min Z-score for sustained Zero Sequence Voltage
I0_PERSIST_Z = 3.0  # Min Z-score for sustained Zero Sequence Current

MAX_TRANSIENT_WINDOWS = 3  # Max duration for events classified as transient

# -----------------------------------------------------------------------------
# Detection Logic Thresholds
# -----------------------------------------------------------------------------
BASELINE_SECONDS = 0.25  # Duration of the pre-fault learning window
MIN_BASELINE_WINDOWS = 5  # Minimum samples required for statistical stability

# Protection Element Thresholds
MIN_OP_CURRENT_PU = 0.05  # Minimum Current (PU) to validate ratio calculations
VOLTAGE_SAG_THRESH = 0.92  # Voltage (PU) threshold for Sag detection (ANSI 27)

# Physical Constraint Thresholds (Noise Floors)
R2_ABS_THRESHOLD = 0.12  # Absolute Floor: Negative Sequence Ratio (|I2|/|I1|)
R0_ABS_THRESHOLD = 0.05  # Absolute Floor: Zero Sequence Ratio (|I0|/|I1|)
DELTA_I_THRESHOLD = 1.3  # Minimum Overcurrent Ratio (I_measured / I_baseline)
DELTA_U1_SAG_THRESHOLD = 0.95  # Max Positive Seq Voltage ratio for sag flagging

# Relay-Cleared Fault Parameters (Breaker Operation Detection)
RELAY_CLEARED_MAX_CYCLES = 50  # Max fault duration before clearing implies failure
RELAY_TRIP_I_RATIO = (
    0.25  # Post-trip current ratio (<25% baseline implies open breaker)
)
RELAY_INCEPTION_R0 = 0.05  # Min asymmetry required at inception
RELAY_INCEPTION_DI = 1.2  # Min current jump at inception
RELAY_CLEARED_V0_RATIO = 0.20

# Statistical Anomaly Thresholds (Z-Scores)
# A Z-Score of 3.0 corresponds to ~99.7% confidence interval (assuming normality).
FFT_HF_START_MULT = 3  # HF analysis start harmonic (3rd harmonic = 150Hz)
FFT_Z_THRESHOLD_MIN = 3.0  # Threshold for High-Frequency distortion
THD_Z_THRESHOLD_MIN = 3.0  # Threshold for Total Harmonic Distortion
WAVELET_Z_THRESHOLD_MIN = 3.0  # Threshold for Wavelet Energy anomalies
DRMS_Z_THRESHOLD_MIN = 3.0  # Threshold for dRMS/dt anomalies

# Fusion and Decision Logic
MIN_CONSEC_FAULT_WINDOWS = 2  # Minimum consecutive anomalies to flag a candidate
FUSION_SCORE_THRESHOLD = 2  # Base sensitivity weight

# -----------------------------------------------------------------------------
# Dataset Channel Mapping
# -----------------------------------------------------------------------------
V_IDX = [0, 1, 2]  # Phase Voltages (Va, Vb, Vc)
I_IDX = [3, 4, 5]  # Phase Currents (Ia, Ib, Ic)

# Dataset quantization steps (ADC counts -> physical units)
V_STEP_VOLTS = 18.310  # Volts per count
I_STEP_AMPS = 4.314  # Amps per count

# =============================================================================
# DATA ACQUISITION & INITIALIZATION
# =============================================================================

print(f"Loading DATA_S from: {DATA_S_PATH} ...")
if os.path.exists(DATA_S_PATH):
    DATA_S = np.load(DATA_S_PATH)["DATA_S"]  # Shape: (n_events, 6, n_samples)
    MAX_SAMPLES = 12053
    if DATA_S.shape[0] > MAX_SAMPLES:
        print(f"Subsampling: Processing first {MAX_SAMPLES} events.")
        DATA_S = DATA_S[:MAX_SAMPLES]
    N_EVENTS, N_CHANNELS, N_SAMPLES = DATA_S.shape
    print(f"Dataset initialized with shape: {DATA_S.shape}")
else:
    print(f"ERROR: Resource not found at {DATA_S_PATH}")
    # Resilience: Initialize dummy tensor to prevent crash during dry runs
    DATA_S = np.zeros((1, 6, 1000))
    N_EVENTS, N_CHANNELS, N_SAMPLES = DATA_S.shape


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================


def demean_and_detrend(x: np.ndarray, ma_win: int = MA_WINDOW) -> np.ndarray:
    """
    Removes DC offsets and suppresses low-frequency drift.

    Rationale:
        Field recordings often contain static offsets (ADC bias) or thermal drift.
        Removal is critical to ensure Zero Sequence components reflect physical
        ground faults rather than measurement artifacts.

    Args:
        x (np.ndarray): Input 1D raw signal.
        ma_win (int): Kernel size for moving-average trend estimation.

    Returns:
        np.ndarray: Zero-mean, detrended signal.
    """
    x = x.astype(float)
    x = x - np.mean(x)  # Remove static DC offset

    if ma_win > 1:
        # Estimate low-frequency non-linear drift via uniform moving average
        kernel = np.ones(ma_win) / ma_win
        trend = np.convolve(x, kernel, mode="same")
        x = x - trend

    return x


def window_signal(event_6ch: np.ndarray, win_size: int = CYCLE_SAMPLES) -> np.ndarray:
    """
    Partitions continuous waveforms into discrete, synchronous single-cycle windows.

    Rationale:
        Power system metrics (RMS, Symmetrical Components) are defined over periodic
        cycles. Synchronous windowing allows for high-resolution tracking of
        fault evolution cycle-by-cycle.

    Args:
        event_6ch (np.ndarray): Raw data (6, n_samples).
        win_size (int): Samples per window (128 for 50Hz @ 6.4kHz).

    Returns:
        np.ndarray: Reshaped array (n_windows, 6, win_size).
    """
    _, n = event_6ch.shape
    n_windows = n // win_size

    # Truncate trailing samples that do not form a complete cycle
    trimmed = event_6ch[:, : n_windows * win_size]

    # Reshape to (Channels, Windows, Samples)
    windows = trimmed.reshape(6, n_windows, win_size)

    # Transpose to (Windows, Channels, Samples) for iteration
    return np.transpose(windows, (1, 0, 2))


# =============================================================================
# FEATURE EXTRACTION ENGINE
# =============================================================================


def compute_rms_per_phase(win: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Root Mean Square (RMS) magnitude for individual phases.

    Args:
        win (np.ndarray): Single-cycle window (6, 128).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (V_rms, I_rms) for phases [A, B, C].
    """
    V = win[V_IDX]
    I = win[I_IDX]
    Vrms = np.sqrt(np.mean(V**2, axis=1))
    Irms = np.sqrt(np.mean(I**2, axis=1))
    return Vrms, Irms


def compute_symmetrical_components(
    three_phase_win: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Performs Fortescue Transformation to decompose the three-phase system into
    Symmetrical Components.

    Components:
        - I1 (Positive): Balanced load flow.
        - I2 (Negative): Indicator of unbalance (Line-to-Line faults).
        - I0 (Zero): Indicator of ground leakage (Line-to-Ground faults).

    Args:
        three_phase_win (np.ndarray): 3-phase cycle data.

    Returns:
        Tuple[float, float, float]: RMS magnitudes of (Zero, Positive, Negative) sequences.
    """
    # Phasor rotation operator alpha (120-degree shift)
    alpha = np.exp(2j * np.pi / 3)

    # Symmetrical Component Transformation Matrix
    T = (1 / 3) * np.array(
        [
            [1, 1, 1],  # Zero Sequence row
            [1, alpha**2, alpha],  # Positive Sequence row
            [1, alpha, alpha**2],  # Negative Sequence row
        ],
        dtype=complex,
    )

    # Transform Time-Domain -> Sequence-Domain
    comp012 = T @ three_phase_win
    c0, c1, c2 = comp012

    # Compute scalar RMS magnitudes
    comp0_rms = np.sqrt(np.mean(np.abs(c0) ** 2))
    comp1_rms = np.sqrt(np.mean(np.abs(c1) ** 2))
    comp2_rms = np.sqrt(np.mean(np.abs(c2) ** 2))

    return comp0_rms, comp1_rms, comp2_rms


def compute_fft_features(win: np.ndarray, fs: float = FS) -> Tuple[float, float]:
    """
    Extracts frequency-domain descriptors to identify non-linear fault signatures.

    Metrics:
        1. HF Energy Ratio: Energy > 150 Hz (indicative of arcing/transients).
        2. THD: Total Harmonic Distortion of the current.

    Args:
        win (np.ndarray): Current cycle data.
        fs (float): Sampling rate.

    Returns:
        Tuple[float, float]: (High-Frequency Ratio, THD).
    """
    I = win[I_IDX]  # Analyze Current channels
    n = I.shape[1]
    eps = 1e-12

    # Real FFT
    spec = np.fft.rfft(I, axis=1)  # Shape: (3, n_bins)
    power = np.abs(spec) ** 2  # Power spectrum

    # Map fundamental frequency (50 Hz) to bin index
    fund_bin = int(round(F_NOM * n / fs))
    fund_bin = max(fund_bin, 1)

    # 1. High Frequency Ratio
    # Sum energy from 3rd harmonic (150 Hz) upwards
    hf_start_bin = FFT_HF_START_MULT * fund_bin
    hf_power = np.sum(power[:, hf_start_bin:])
    total_power = np.sum(power) + eps
    hf_ratio = float(hf_power / total_power)

    # 2. Total Harmonic Distortion (THD)
    amp = np.abs(spec)

    # Fundamental energy (integrated over small band to handle freq deviation)
    fund_band = slice(fund_bin - 1 if fund_bin > 1 else fund_bin, fund_bin + 2)
    I1_mag_sq = np.sum(amp[:, fund_band] ** 2)

    # Harmonic energy (all components >= 2nd harmonic)
    harm_start = 2 * fund_bin
    harm_mag_sq = np.sum(amp[:, harm_start:] ** 2)

    thd = float(np.sqrt(harm_mag_sq) / (np.sqrt(I1_mag_sq) + eps))

    return hf_ratio, thd


def compute_wavelet_energy(win: np.ndarray) -> float:
    """
    Quantifies transient energy using Discrete Wavelet Transform (DWT).
    Wavelets offer superior time-frequency localization for impulsive events compared to FFT.

    Args:
        win (np.ndarray): Current cycle data.

    Returns:
        float: Mean energy of Detail Coefficients. NaN if PyWavelets is missing.
    """
    if not HAS_PYWT:
        return np.nan

    I = win[I_IDX]
    energies = []

    for ch in range(I.shape[0]):
        # Multilevel decomposition (Daubechies 2, level 3)
        coeffs = pywt.wavedec(I[ch], "db2", level=3)

        # Sum energy of Detail Coefficients (high-freq components)
        detail_coeffs = coeffs[1:]
        e = 0.0
        for c in detail_coeffs:
            e += float(np.sum(c**2))
        energies.append(e)

    return float(np.mean(energies))


def fast_ground_fault_detector(
    raw_event: np.ndarray, fs: float, t_max: float = 0.25
) -> Optional[float]:
    """
    Fast-path detector for Line-to-Ground faults occurring within the first cycle.
    Operates on raw sub-cycle data before the main cycle-synchronous windowing loop
    to minimize detection latency for severe ground faults.

    Args:
        raw_event (np.ndarray): The raw 6-channel event waveform.
        fs (float): Sampling frequency.
        t_max (float): Maximum time to search within the event (seconds).

    Returns:
        Optional[float]: Inception time in seconds, or None if no fast-path trigger is found.
    """
    Va, Vb, Vc = raw_event[0], raw_event[1], raw_event[2]
    Ia, Ib, Ic = raw_event[3], raw_event[4], raw_event[5]

    n_max = int(t_max * fs)

    # Instantaneous Residuals
    u0 = (Va + Vb + Vc) / 3.0
    i0 = (Ia + Ib + Ic) / 3.0

    # Derivative of Zero Sequence Current
    di0 = np.diff(i0, prepend=i0[0]) * fs

    # -------------------------------
    # Absolute Physical Thresholds
    # -------------------------------
    I0_MIN = 0.05 * np.max(np.abs(Ia)) + 1e-3
    U0_MIN = 0.05 * np.max(np.abs(Va)) + 1e-3
    DI0_MIN = 5.0 * np.std(di0[:n_max])

    FAST_WIN = int(0.005 * fs)  # 5 ms sliding window

    for k in range(FAST_WIN, n_max):
        # Trigger: Significant residual current AND (current jump OR voltage residual)
        if np.max(np.abs(i0[k - FAST_WIN : k])) > I0_MIN and (
            np.max(np.abs(di0[k - FAST_WIN : k])) > DI0_MIN
            or np.max(np.abs(u0[k - FAST_WIN : k])) > U0_MIN
        ):
            return k / fs  # Inception time (seconds)

    return None


def compute_event_features(windows: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Orchestrates feature extraction for an entire event record cycle-by-cycle.

    Args:
        windows (np.ndarray): Cycle-partitioned event data.

    Returns:
        Dict[str, np.ndarray]: Time-series mapping of extracted features.
    """
    n_windows = windows.shape[0]

    # Pre-allocation for performance
    Vrms_total = np.zeros(n_windows)
    Irms_total = np.zeros(n_windows)
    V0, V1, V2 = np.zeros(n_windows), np.zeros(n_windows), np.zeros(n_windows)
    I0, I1, I2 = np.zeros(n_windows), np.zeros(n_windows), np.zeros(n_windows)
    U0_rms = np.zeros(n_windows)
    I0_rms = np.zeros(n_windows)
    fft_hf_ratio = np.zeros(n_windows)
    thd = np.zeros(n_windows)
    wavelet_energy = np.zeros(n_windows)

    # Iterate through all windows
    for w in range(n_windows):
        win = windows[w]

        # RMS
        Vrms, Irms = compute_rms_per_phase(win)
        Vrms_total[w] = np.sqrt(np.sum(Vrms**2))
        Irms_total[w] = np.sqrt(np.sum(Irms**2))

        # Symmetrical Components
        V0_w, V1_w, V2_w = compute_symmetrical_components(win[V_IDX])
        I0_w, I1_w, I2_w = compute_symmetrical_components(win[I_IDX])

        V0[w], V1[w], V2[w] = V0_w, V1_w, V2_w
        I0[w], I1[w], I2[w] = I0_w, I1_w, I2_w

        U0_rms[w] = V0_w
        I0_rms[w] = I0_w

        # Frequency Domain
        hf_w, thd_w = compute_fft_features(win)
        fft_hf_ratio[w] = hf_w
        thd[w] = thd_w

        # Time-Frequency Domain
        wavelet_energy[w] = compute_wavelet_energy(win)

    return {
        "Vrms_total": Vrms_total,
        "Irms_total": Irms_total,
        "V0": V0,
        "V1": V1,
        "V2": V2,
        "I0": I0,
        "I1": I1,
        "I2": I2,
        "U0_rms": U0_rms,
        "I0_rms": I0_rms,
        "fft_hf_ratio": fft_hf_ratio,
        "thd": thd,
        "wavelet_energy": wavelet_energy,
    }


# =============================================================================
# FAULT DETECTION LOGIC (FUSION ENGINE)
# =============================================================================


def detect_fault_start(
    features: Dict[str, np.ndarray],
    raw_event: np.ndarray,
    win_size: int = CYCLE_SAMPLES,
    fs: float = FS,
    baseline_seconds: float = BASELINE_SECONDS,
    min_baseline_windows: int = MIN_BASELINE_WINDOWS,
    min_consec_fault_windows: int = MIN_CONSEC_FAULT_WINDOWS,
) -> Tuple[Optional[float], np.ndarray, Dict[str, Any]]:
    """
    Executes the multi-variate Fusion Engine to determine fault inception.

    Workflow:
        1. Baseline Learning: Compute robust statistics (Median/MAD) for pre-fault state.
        2. Normalization: Convert raw features to Z-Scores resilient to masking.
        3. Logic Fusion: Combine supervised physical detectors (voltage sag, unbalance)
           with statistical anomalies.
        4. Gating: Apply hysteresis and persistence checks to reject non-fault transients.

    Args:
        features (Dict): Time-series feature vectors.
        raw_event (np.ndarray): Original raw signal (used for fast-path).
        win_size (int): Samples per window.
        fs (float): Sampling frequency.
        baseline_seconds (float): Length of learning window.
        min_baseline_windows (int): Min samples for baseline validity.
        min_consec_fault_windows (int): Persistence requirement.

    Returns:
        Tuple: (fault_start_s, binary_fault_mask, detection_metadata_dict).
    """
    # ==================================================
    # Fast-Path Sub-Cycle Line-to-Ground Fault Detector
    # ==================================================
    EDGE_MARGIN_S = 0.06  # or 0.02 (one cycle)
    fast_t = fast_ground_fault_detector(
        raw_event, fs, t_max=baseline_seconds + EDGE_MARGIN_S
    )

    # Retain the fast-path trigger timestamp for subsequent fusion and metadata logging.
    fast_start_win = None
    if fast_t is not None:
        fast_start_win = int(fast_t * fs // win_size)

    # Unpack time-series features
    I1 = features["I1"]
    I2 = features["I2"]
    I0 = features["I0"]
    U0 = features["U0_rms"]
    Itot = features["Irms_total"]
    V1 = features["V1"]
    fft_hf_ratio = features["fft_hf_ratio"]
    thd = features["thd"]
    wavelet_energy = features["wavelet_energy"]

    n_windows = I1.shape[0]
    eps = 1e-12

    # Initialize logging variables (safe for early-return execution paths)
    fast_path_activated = 0
    veto_loadpickup_activated = 0
    veto_resonance_activated = 0

    # -------------------------------------------------------------------------
    # 1. Robust Baseline Estimation (Pre-fault Statistical Learning)
    # -------------------------------------------------------------------------
    n_prefault = max(
        min_baseline_windows,
        int(baseline_seconds * fs) // win_size,
    )
    n_prefault = min(n_prefault, n_windows)
    baseline_slice = slice(0, n_prefault)

    def robust_mean_std(x: np.ndarray) -> Tuple[float, float]:
        """Calculates robust statistics (Median & MAD) to reject outliers."""
        xb = x[baseline_slice]
        med = float(np.median(xb))
        mad = float(np.median(np.abs(xb - med))) + eps
        # Scale factor (1.4826) approximates the standard deviation for a normal distribution.
        robust_std = 1.4826 * mad
        return med, robust_std

    Itot_base_mean, Itot_base_std = robust_mean_std(Itot)
    V1_base_mean, V1_base_std = robust_mean_std(V1)

    # ============================
    # System Constraint: Abort processing if pre-fault conditions indicate a de-energized line.
    if V1_base_mean < 1e-2:
        fault_flag = np.zeros(n_windows, dtype=int)
        detection_info = {
            "fusion_score_at_start": 0.0,
            "flag_unbalance": 0,
            "flag_negseq": 0,
            "flag_zeroseq": 0,
            "flag_current": 0,
            "flag_voltage": 0,
            "flag_fft": 0,
            "flag_thd": 0,
            "flag_wavelet": 0,
            "flag_drms": 0,
            "fast_path_activated": 0,
            "veto_loadpickup_activated": 0,
            "veto_resonance_activated": 0,
        }
        return None, fault_flag, detection_info

    deenergized_or_lightload = Itot_base_mean < 1e-2

    # Relative Delta Features (normalized against the baseline)
    Itot_baseline = Itot_base_mean + eps
    V1_baseline = V1_base_mean + eps
    delta_I = Itot / Itot_baseline
    delta_U1 = V1 / V1_baseline

    # Identify energization signatures characterized by simultaneous voltage and current rise.
    energization_like = (delta_U1 > 1.10) & (delta_I > 1.10)

    # Compute normalized symmetrical components
    I1_base_mean, _ = robust_mean_std(I1)
    I1_floor = max(0.10 * abs(I1_base_mean), 1e-2)
    I1_safe = np.where(np.abs(I1) < I1_floor, I1_floor, I1)

    R2 = I2 / I1_safe  # Negative Sequence Ratio (Asymmetry)
    R0 = I0 / I1_safe  # Zero Sequence Ratio (Ground Leakage)

    # Execute comprehensive statistical profiling
    R2_mean, R2_std = robust_mean_std(R2)
    R0_mean, R0_std = robust_mean_std(R0)
    dI_mean, dI_std = robust_mean_std(delta_I)
    dU1_mean, dU1_std = robust_mean_std(delta_U1)
    fft_mean, fft_std = robust_mean_std(fft_hf_ratio)
    thd_mean, thd_std = robust_mean_std(thd)

    if np.all(np.isnan(wavelet_energy[baseline_slice])):
        wavelet_mean, wavelet_std = 0.0, 1.0
    else:
        wavelet_mean, wavelet_std = robust_mean_std(wavelet_energy)

    # Calculate the differential feature: Rate of Change of Current (dRMS/dt)
    dItot = np.zeros_like(Itot)
    dItot[1:] = np.diff(Itot) / (win_size / fs)
    dItot_mean, dItot_std = robust_mean_std(dItot)

    # -------------------------------------------------------------------------
    # 2. Z-Score Normalization (Anomalous Statistical Departure)
    # -------------------------------------------------------------------------
    z_R2 = (R2 - R2_mean) / (R2_std + eps)
    z_R0 = (R0 - R0_mean) / (R0_std + eps)
    z_delta_I = (delta_I - dI_mean) / (dI_std + eps)
    z_delta_U1 = (delta_U1 - dU1_mean) / (dU1_std + eps)
    z_fft = (fft_hf_ratio - fft_mean) / (fft_std + eps)
    z_thd = (thd - thd_mean) / (thd_std + eps)

    # --- Ground-Channel Z-Scores (critical for Line-to-Ground fault detection) ---
    I0_mean, I0_std = robust_mean_std(I0)
    U0_mean, U0_std = robust_mean_std(U0)

    z_I0 = (I0 - I0_mean) / (I0_std + eps)
    z_U0 = (U0 - U0_mean) / (U0_std + eps)

    if np.all(np.isnan(wavelet_energy)):
        z_wavelet = np.zeros_like(Itot)
    else:
        z_wavelet = (wavelet_energy - wavelet_mean) / (wavelet_std + eps)

    z_dItot = (dItot - dItot_mean) / (dItot_std + eps)

    # Validation Helper Functions
    TRANSIENT_MAX_WINDOWS = MAX_TRANSIENT_WINDOWS
    TRANSIENT_MIN_WINDOWS = 1
    TRANSIENT_DECAY_WINDOWS = 6
    TRANSIENT_R0_STRONG = R0_ABS_THRESHOLD
    TRANSIENT_ZR0_STRONG = 3.0
    TRANSIENT_RESTORE_FACTOR = 1.5

    def is_strong_ground_impulse(w: int) -> bool:
        """Checks for high-magnitude ground asymmetry anomalies."""
        return (R0[w] > TRANSIENT_R0_STRONG) and (z_R0[w] > TRANSIENT_ZR0_STRONG)

    def looks_like_inrush(k0: int) -> bool:
        """Monotonic Decay Check: Identifies inrush-like exponential decay."""
        k1 = min(k0 + 6, n_windows)
        if k1 <= k0 + 2:
            return False
        return bool(np.all(np.diff(Itot[k0:k1]) < 0))

    inrush_like = np.zeros_like(Itot, dtype=bool)
    for k in range(n_windows - 6):
        if looks_like_inrush(k):
            inrush_like[k] = True

    def decays_quickly(k0: int, k_end: int) -> bool:
        """Validates that a signal returns to baseline post-transient."""
        k2 = min(k_end + TRANSIENT_DECAY_WINDOWS, n_windows)
        if k_end >= k2:
            return False
        baseline_level = max(
            R0_mean + TRANSIENT_RESTORE_FACTOR * R0_std, R0_ABS_THRESHOLD * 0.5
        )
        post = R0[k_end:k2]
        near = np.sum(post < baseline_level)
        return bool(near >= int(0.7 * len(post)))

    def passes_persistence_gate(k0: int) -> bool:
        """Ensures physical indicators sustain across the persistence window."""
        k1 = min(k0 + PERSISTENCE_WINDOWS, n_windows)
        phys_count = sum(
            1
            for k in range(k0, k1)
            if (
                R0[k] > R0_PERSIST_THRESHOLD
                or R2[k] > R2_PERSIST_THRESHOLD
                or delta_I[k] > DELTA_I_THRESHOLD
            )
        )
        return bool(phys_count >= int(0.75 * PERSISTENCE_WINDOWS))

    def valid_physics_start(w: int) -> bool:
        """Confirms that fault inception correlates with growing electrical asymmetry."""
        return (
            (R2[w] > R2_ABS_THRESHOLD and z_R2[w] > 3)
            or (R0[w] > R0_ABS_THRESHOLD and z_R0[w] > 3)
            or (delta_I[w] > DELTA_I_THRESHOLD and z_delta_I[w] > 3)
        )

    def passes_steady_state_fault(k0: int) -> bool:
        """Validates the steady-state characteristics of a confirmed fault."""
        settle_start, settle_end = k0 + 3, min(k0 + 10, n_windows)
        if settle_end <= settle_start:
            return False
        mean_r2, mean_r0 = np.mean(R2[settle_start:settle_end]), np.mean(
            R0[settle_start:settle_end]
        )
        mean_dI, mean_dU = np.mean(delta_I[settle_start:settle_end]), np.mean(
            delta_U1[settle_start:settle_end]
        )
        return bool(
            (mean_r2 > 0.12 or mean_r0 > 0.05) or (mean_dI > 1.3 and mean_dU < 0.97)
        )

    def is_energization_settle(k0: int) -> bool:
        """
        Filters Load Pickup / Energization events.

        Logic:
        - Requires absence of sustained voltage sag after the candidate window.
        - Ensures unbalance (R0/R2) is not sustained post-settling.
        - Utilizes fraction-of-windows analysis for robust outlier handling.
        """
        s1 = min(k0 + 3, n_windows)  # Bypass the initial switching transient
        s2 = min(
            k0 + 25, n_windows
        )  # Evaluate across an extended monitoring window (~0.5s)

        if s2 <= s1:
            return False

        # Verify no-sag condition via robust quantile assessment
        dU = delta_U1[s1:s2]
        no_sag = (
            float(np.quantile(dU, 0.2)) > 0.97
        )  # Ensures the 20th percentile remains near nominal voltage

        # Evaluate persistence of electrical unbalance
        r0 = R0[s1:s2]
        r2 = R2[s1:s2]
        frac_unbalance = float(
            np.mean((r0 > R0_ABS_THRESHOLD) | (r2 > R2_ABS_THRESHOLD))
        )

        # Confirm presence of load current (ensures breaker remains closed)
        dI = delta_I[s1:s2]
        current_present = float(np.quantile(dI, 0.3)) > 1.05

        # Rejection Signature: Healthy voltage, active current, and transient unbalance.
        return no_sag and current_present and (frac_unbalance < 0.35)

    def has_physical_retrigger(w: int) -> bool:
        """Ensures late-stage triggers exhibit legitimate electrical asymmetry."""
        return R2[w] > R2_ABS_THRESHOLD or R0[w] > R0_ABS_THRESHOLD

    # -------------------------------------------------------------------------
    # 3. Decision Fusion & Heuristic Masking
    # -------------------------------------------------------------------------

    # Robust load masking prevents false triggers on noise under near-zero baseline conditions.
    Itot_min = max(Itot_base_mean + 5 * Itot_base_std, 0.05)
    load_mask = Itot > Itot_min

    # -------------------------------------------------------------------------
    # SECTION: Magnitude Supervision, Load Encroachment & Resonance Blocker
    # -------------------------------------------------------------------------

    # 0. Define a robust current reference to avoid "zero-load" division errors.
    # Apply a hard operational floor to pre-fault current estimates during de-energized
    # states to prevent noise-induced triggering and divide-by-zero anomalies.
    I_REF = max(I1_base_mean, 0.5)

    # 1. Quantify Voltage Unbalance (U0 Ratio)
    U0_ratio = U0 / (V1_base_mean + eps)

    # 2. Resonance Blocker (Current-Dominance Validation)
    # High voltage unbalance coupled with weak residual current typically indicates
    # resonance or VT saturation. Genuine ground faults remain current-dominant.
    is_resonance_likely = (U0_ratio > 0.05) & (I0 < (I_REF * 0.15))

    # 3. Magnitude Supervision (ANSI 50 supervising ANSI 46)
    # Maintain high sensitivity (5%) for negative-sequence current to detect broken conductors.
    # Ensure zero-sequence current is substantial and distinct from voltage-driven resonance.
    MIN_OP_I2_PU = 0.05
    MIN_OP_I0_PU = 0.05

    # Leverage the reference current to prevent false trips caused by capacitive charging currents.
    is_substantial_I2 = I2 > (I_REF * MIN_OP_I2_PU)

    # Validate I0 based on substantial magnitude and absence of resonance characteristics.
    is_substantial_I0 = (I0 > (I_REF * MIN_OP_I0_PU)) & (~is_resonance_likely)

    # ---------------------------------------------------------------------
    # Line-to-Ground (L-G) Fault Condition: Enables detection irrespective of voltage sag.
    # Evaluates strong residual quantities and zero-sequence ratios under magnitude supervision.
    # ---------------------------------------------------------------------
    # --- Ground Fault Operating Modes ---
    # Mode 1: Impulsive / Transient L-G Faults (Current-Dominant)
    cond_ground_impulse = (
        ((z_I0 > 5.0) | (z_R0 > 4.0)) & is_substantial_I0 & (z_I0 > z_U0)
    )

    # Mode 2: Sustained L-G Faults (Ratio-Dominant)
    cond_ground_block = (
        R0 > 0.10
    ) & is_substantial_I0  # Apply a stringent absolute threshold

    cond_ground = cond_ground_impulse | cond_ground_block

    # Enforce persistence of the ground fault condition across consecutive windows.
    # Mode-aware ground persistence evaluation
    cond_ground_persistent = np.zeros_like(cond_ground, dtype=bool)

    for k in range(n_windows - 1):
        # Case 1: Impulsive transients permit single-window detection.
        if cond_ground_impulse[k]:
            cond_ground_persistent[k] = True
        # Case 2: Sustained faults require corroboration over consecutive windows.
        elif cond_ground_block[k] and cond_ground_block[k + 1]:
            cond_ground_persistent[k] = True

    # 4. Voltage Supervision (ANSI 27 load pickup blocker)
    VOLT_HEALTHY_THRESH = 0.92
    is_voltage_healthy = delta_U1 > VOLT_HEALTHY_THRESH

    # 5. Update Detectors with Magnitude and Voltage Supervision
    cond_unbalance = (((R2 > R2_ABS_THRESHOLD) | (z_R2 > 4.0)) & is_substantial_I2) | (
        ((R0 > R0_ABS_THRESHOLD) | (z_R0 > 4.0)) & is_substantial_I0
    )

    # --- Observability and Logging Metrics (Non-Operational) ---
    cond_negseq = ((R2 > R2_ABS_THRESHOLD) | (z_R2 > 4.0)) & is_substantial_I2
    cond_zeroseq = ((R0 > R0_ABS_THRESHOLD) | (z_R0 > 4.0)) & is_substantial_I0

    # Track decision-path veto activations for post-event analysis.
    veto_resonance_activated = int(
        np.any(is_resonance_likely[load_mask])
    )  # Indicates resonance blocker activation
    veto_loadpickup_activated = int(
        np.any(inrush_like[load_mask])
    )  # Indicates inrush or load pickup signature detection

    # Initialize fast-path flag (set operationally within the fast-path return branch)
    fast_path_activated = 0

    # Voltage Sag Detection (ANSI 27)
    cond_voltage_sag = (delta_U1 < DELTA_U1_SAG_THRESHOLD) | (z_delta_U1 < -4.0)

    # Overcurrent Detection with Load Pickup Constraint (ANSI 50)
    raw_current_trigger = (delta_I > max(DELTA_I_THRESHOLD, 1.5)) | (z_delta_I > 4.0)
    cond_current = raw_current_trigger & (~is_voltage_healthy)

    # -------------------------------------------------------------------------
    # SECTION: Logic Integration & Decision Engine
    # -------------------------------------------------------------------------

    cond_fft = (z_fft > max(FFT_Z_THRESHOLD_MIN, 4.0)) & (
        fft_hf_ratio > fft_mean + 3.0 * fft_std
    )
    cond_thd = (z_thd > max(THD_Z_THRESHOLD_MIN, 4.0)) & (
        thd > thd_mean + 3.0 * thd_std
    )

    if np.all(np.isnan(wavelet_energy)):
        cond_wavelet = np.zeros_like(Itot, dtype=bool)
    else:
        cond_wavelet = (z_wavelet > max(WAVELET_Z_THRESHOLD_MIN, 4.0)) & (
            wavelet_energy > wavelet_mean + 3.0 * wavelet_std
        )

    cond_dItot = np.abs(z_dItot) > max(DRMS_Z_THRESHOLD_MIN, 4.0)

    # --- Integration of Supervised Logic into the Fusion Decision Engine ---
    # 1. Update core logic definitions with supervised conditions.
    # Integrating supervised triggers as the core condition ensures that
    # voltage blocking (e.g., Load Encroachment) correctly restricts hard trips.
    core_unbalance = cond_unbalance
    core_current = cond_current
    core_voltage = cond_voltage_sag

    # Strict Gating: Demands simultaneous overcurrent and either unbalance or voltage sag.
    core_condition = (
        cond_ground_persistent & (cond_dItot | raw_current_trigger) & (~inrush_like)
    ) | (cond_current & (cond_unbalance | cond_voltage_sag) & (~inrush_like))

    # 2. Configure the Weighted Voting Matrix
    weights = {
        "unbalance": 2.0,
        "current": 2.0,
        "voltage": 2.0,
        "fft": 1.0,
        "thd": 1.0,
        "wavelet": 1.0,
        "drms": 1.0,
    }

    # Ensure integration of supervised logic components ('cond_' variables).
    detectors = {
        "unbalance": cond_unbalance,
        "current": cond_current,
        "voltage": cond_voltage_sag,
        "fft": cond_fft,
        "thd": cond_thd,
        "wavelet": cond_wavelet,
        "drms": cond_dItot,
    }

    fusion_score = np.zeros(n_windows, dtype=float)
    for name, cond in detectors.items():
        fusion_score += weights[name] * cond.astype(float)

    # Consolidate fast-path inception results with corresponding fusion metrics and flags.
    if fast_start_win is not None:
        start_win = int(np.clip(fast_start_win, 0, n_windows - 1))
        fault_flag = np.zeros(n_windows, dtype=int)
        fault_flag[start_win:] = 1

        fast_path_activated = 1  # Logging metric

        detection_info = {
            "fusion_score_at_start": float(fusion_score[start_win]),
            "flag_unbalance": int(cond_unbalance[start_win]),
            "flag_negseq": int(cond_negseq[start_win]),
            "flag_zeroseq": int(cond_zeroseq[start_win]),
            "flag_current": int(cond_current[start_win]),
            "flag_voltage": int(cond_voltage_sag[start_win]),
            "flag_fft": int(cond_fft[start_win]),
            "flag_thd": int(cond_thd[start_win]),
            "flag_wavelet": int(cond_wavelet[start_win]),
            "flag_drms": int(cond_dItot[start_win]),
            "fast_path_activated": int(fast_path_activated),
            "veto_loadpickup_activated": int(veto_loadpickup_activated),
            "veto_resonance_activated": int(veto_resonance_activated),
        }

        return fast_t, fault_flag, detection_info

    # Candidate Identification via Sensitivity Scaling
    # Apply supervision to high-sensitivity triggers to mitigate resonance false positives.
    # Replaces unconditional z_R0 triggers with a strict substantial current requirement.
    safe_z_R0_trigger = (z_R0 > 5.0) & is_substantial_I0

    high_sensitivity_candidate = load_mask & (
        (fusion_score >= 3.0) | cond_dItot | cond_wavelet | safe_z_R0_trigger
    )

    FUSION_SCORE_STRICT = 5.0
    fault_candidate = load_mask & core_condition & (fusion_score >= FUSION_SCORE_STRICT)

    # Transient logic evaluation incorporating the resonance blocker constraint.
    # Substitutes raw unbalance ratios with the fully supervised unbalance condition.
    transient_candidate = (
        load_mask
        & (~energization_like)
        & cond_unbalance
        & (cond_dItot | cond_wavelet)
        & (~inrush_like)
    )

    # Sequential Validation of Fault Candidate Persistence Runs
    w = n_prefault
    while w < n_windows:
        if fault_candidate[w]:
            if not valid_physics_start(w):
                w0 = w
                while w < n_windows and fault_candidate[w]:
                    fault_candidate[w] = False
                    w += 1
                continue
        w += 1

    # Artifact Suppression (Pre-fault baseline and post-event de-energization)
    early_mask = np.arange(n_windows) < n_prefault
    strong_early_fault = (delta_I > 1.5) & (delta_U1 < 0.95)
    fault_candidate[early_mask & ~strong_early_fault] = False

    Itot_tail, V1_tail = Itot / (Itot_baseline + eps), V1 / (V1_baseline + eps)
    end_artifact = (Itot_tail < 0.1) & (np.abs(V1_tail - 1.0) < 0.05)
    fault_candidate[end_artifact] = False

    # -------------------------------------------------------------------------
    # 4. Final Event Classification (Sustained, Relay-Cleared, or Transient)
    # -------------------------------------------------------------------------
    fault_flag = fault_candidate.astype(int)
    start_win, run, candidate_start = None, 0, None

    # Classification Path 1: Sustained Faults
    for w in range(0, n_windows):
        if fault_candidate[w]:
            run += 1
            if run == min_consec_fault_windows:
                candidate_start = w - run + 1
            if candidate_start is not None and run >= min_consec_fault_windows:
                if is_energization_settle(candidate_start):
                    kk = candidate_start
                    while kk < n_windows and fault_candidate[kk]:
                        fault_candidate[kk] = False
                        high_sensitivity_candidate[kk] = False
                        kk += 1
                    run, candidate_start = 0, None
                    continue

                k1 = min(candidate_start + 5, n_windows)
                if not passes_persistence_gate(candidate_start):
                    break
                if not passes_steady_state_fault(candidate_start):
                    break
                if run < PERSISTENCE_WINDOWS:
                    continue
                start_win = candidate_start
                break
        else:
            run, candidate_start = 0, None

    if start_win is not None:
        start_time_s = start_win * win_size / fs
        fault_flag[start_win:] = 1

        detection_info = {
            "fusion_score_at_start": float(fusion_score[start_win]),
            "flag_unbalance": int(cond_unbalance[start_win]),
            "flag_negseq": int(cond_negseq[start_win]),
            "flag_zeroseq": int(cond_zeroseq[start_win]),
            "flag_current": int(cond_current[start_win]),
            "flag_voltage": int(cond_voltage_sag[start_win]),
            "flag_fft": int(cond_fft[start_win]),
            "flag_thd": int(cond_thd[start_win]),
            "flag_wavelet": int(cond_wavelet[start_win]),
            "flag_drms": int(cond_dItot[start_win]),
            "fast_path_activated": int(fast_path_activated),
            "veto_loadpickup_activated": int(veto_loadpickup_activated),
            "veto_resonance_activated": int(veto_resonance_activated),
        }

        return start_time_s, fault_flag, detection_info

    # Classification Path 2: Relay-Cleared Faults (Inception followed by current collapse)
    k0 = None
    for w in range(n_prefault - 3, n_windows - 5):
        # Ensure statistical triggers are corroborated by physical current presence.
        is_inception = (fusion_score[w] >= FUSION_SCORE_STRICT) or (
            (z_R0[w] > 8.0) and is_substantial_I0[w]
        )
        if is_inception and k0 is None:
            k0 = w

        if k0 is not None:
            for check_w in range(w, min(w + 3, n_windows)):
                if Itot[check_w] < (Itot_baseline * RELAY_TRIP_I_RATIO):
                    fault_duration = check_w - k0
                    if 1 <= fault_duration <= 60:
                        # Reject energization events that mimic fault pickup before settling.
                        if is_energization_settle(k0):
                            k0 = None
                            break

                        start_time_s = k0 * (win_size / fs)
                        fault_flag[k0 : check_w + 1] = 1
                        start_win = k0  # Identified inception window

                        detection_info = {
                            "fusion_score_at_start": float(fusion_score[start_win]),
                            "flag_unbalance": int(cond_unbalance[start_win]),
                            "flag_negseq": int(cond_negseq[start_win]),
                            "flag_zeroseq": int(cond_zeroseq[start_win]),
                            "flag_current": int(cond_current[start_win]),
                            "flag_voltage": int(cond_voltage_sag[start_win]),
                            "flag_fft": int(cond_fft[start_win]),
                            "flag_thd": int(cond_thd[start_win]),
                            "flag_wavelet": int(cond_wavelet[start_win]),
                            "flag_drms": int(cond_dItot[start_win]),
                            "fast_path_activated": int(fast_path_activated),
                            "veto_loadpickup_activated": int(veto_loadpickup_activated),
                            "veto_resonance_activated": int(veto_resonance_activated),
                        }

                        return start_time_s, fault_flag, detection_info
            if k0 is not None and (w - k0) > 60:
                k0 = None

    # Classification Path 3: Transient / Self-Clearing Faults
    run, k0 = 0, None
    for w in range(n_prefault, n_windows):
        if transient_candidate[w]:
            if run == 0:
                k0 = w
            run += 1
            if run > TRANSIENT_MAX_WINDOWS:
                run, k0 = 0, None
        else:
            if k0 is not None and TRANSIENT_MIN_WINDOWS <= run <= TRANSIENT_MAX_WINDOWS:
                if is_strong_ground_impulse(k0) and decays_quickly(k0, w):
                    start_time_s = k0 * win_size / fs
                    fault_flag[k0:w] = 1
                    start_win = k0

                    detection_info = {
                        "fusion_score_at_start": float(fusion_score[start_win]),
                        "flag_unbalance": int(cond_unbalance[start_win]),
                        "flag_negseq": int(cond_negseq[start_win]),
                        "flag_zeroseq": int(cond_zeroseq[start_win]),
                        "flag_current": int(cond_current[start_win]),
                        "flag_voltage": int(cond_voltage_sag[start_win]),
                        "flag_fft": int(cond_fft[start_win]),
                        "flag_thd": int(cond_thd[start_win]),
                        "flag_wavelet": int(cond_wavelet[start_win]),
                        "flag_drms": int(cond_dItot[start_win]),
                        "fast_path_activated": int(fast_path_activated),
                        "veto_loadpickup_activated": int(veto_loadpickup_activated),
                        "veto_resonance_activated": int(veto_resonance_activated),
                    }
                    return start_time_s, fault_flag, detection_info
            run, k0 = 0, None

    # Default return structure for undetected or benign events
    detection_info = {
        "fusion_score_at_start": None,
        "flag_unbalance": None,
        "flag_negseq": None,
        "flag_zeroseq": None,
        "flag_current": None,
        "flag_voltage": None,
        "flag_fft": None,
        "flag_thd": None,
        "flag_wavelet": None,
        "flag_drms": None,
        "fast_path_activated": None,
        "veto_loadpickup_activated": None,
        "veto_resonance_activated": None,
    }

    return None, fault_flag, detection_info


# =============================================================================
# EXECUTION ORCHESTRATOR
# =============================================================================


def run_fault_detection():
    """
    Primary processing loop. Iterates through the dataset, executes the
    fusion detection engine, and serializes results to a CSV format.
    """
    results = []
    print(f"Executing analytical pipeline on {N_EVENTS} events...")

    for evt_idx in range(N_EVENTS):

        raw_event = DATA_S[evt_idx].astype(float)
        # Convert ADC counts -> physical units
        raw_event[V_IDX, :] *= V_STEP_VOLTS
        raw_event[I_IDX, :] *= I_STEP_AMPS

        # 1. Preprocessing (Detrending)
        detrended = np.zeros_like(raw_event)
        for ch in range(N_CHANNELS):
            detrended[ch] = demean_and_detrend(raw_event[ch], ma_win=MA_WINDOW)

        # 2. Cycle-Synchronous Windowing
        windows = window_signal(detrended, win_size=CYCLE_SAMPLES)

        # 3. High-Dimensional Feature Extraction
        feats = compute_event_features(windows)

        # 4. Fusion Engine Detection
        start_time_s, fault_flag, detection_info = detect_fault_start(
            feats, raw_event=detrended
        )

        results.append(
            {
                "event_idx": evt_idx,
                "fault_start_s": start_time_s,
                "fusion_score_at_start": detection_info["fusion_score_at_start"],
                "flag_unbalance": detection_info["flag_unbalance"],
                "flag_current": detection_info["flag_current"],
                "flag_voltage": detection_info["flag_voltage"],
                "flag_fft": detection_info["flag_fft"],
                "flag_thd": detection_info["flag_thd"],
                "flag_wavelet": detection_info["flag_wavelet"],
                "flag_drms": detection_info["flag_drms"],
                "flag_negseq": detection_info["flag_negseq"],
                "flag_zeroseq": detection_info["flag_zeroseq"],
                "fast_path_activated": detection_info["fast_path_activated"],
                "veto_loadpickup_activated": detection_info[
                    "veto_loadpickup_activated"
                ],
                "veto_resonance_activated": detection_info["veto_resonance_activated"],
            }
        )

        if (evt_idx + 1) % 500 == 0:
            print(f"Log: Processed {evt_idx + 1} / {N_EVENTS} events.")

    # 5. Data Export
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_csv = os.path.join(PROCESSED_DIR, "robust_fault_inception_results.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(
            [
                "event_idx",
                "fault_start_s",
                "fusion_score_at_start",
                "flag_unbalance",
                "flag_negseq",
                "flag_zeroseq",
                "flag_current",
                "flag_voltage",
                "flag_fft",
                "flag_thd",
                "flag_wavelet",
                "flag_drms",
                "fast_path_activated",
                "veto_loadpickup_activated",
                "veto_resonance_activated",
            ]
        )

        # Data rows
        for r in results:
            writer.writerow(
                [
                    r["event_idx"],
                    (
                        round(r["fault_start_s"], 4)
                        if r["fault_start_s"] is not None
                        else ""
                    ),
                    r["fusion_score_at_start"],
                    r["flag_unbalance"],
                    r["flag_negseq"],
                    r["flag_zeroseq"],
                    r["flag_current"],
                    r["flag_voltage"],
                    r["flag_fft"],
                    r["flag_thd"],
                    r["flag_wavelet"],
                    r["flag_drms"],
                    r["fast_path_activated"],
                    r["veto_loadpickup_activated"],
                    r["veto_resonance_activated"],
                ]
            )

    print(f"\nProcessing Complete. Results persisted to: {output_csv}\n")
    return results


if __name__ == "__main__":
    run_fault_detection()
