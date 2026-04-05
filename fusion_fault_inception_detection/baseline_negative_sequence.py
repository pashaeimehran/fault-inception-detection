"""
Overview
--------
Baseline reference implementation using a negative-sequence current threshold
detector for fault inception estimation in the RTE DFR dataset.

This baseline is intentionally simple and interpretable:
1. Convert raw ADC counts to physical units.
2. Demean and detrend all channels.
3. Segment the event into one-cycle windows.
4. Compute symmetrical current components per cycle.
5. Detect the first cycle where the negative-sequence ratio R2 = |I2| / |I1|
   exceeds a robust pre-fault threshold with persistence.

This provides a lightweight physics-based comparator against the full
fusion detector.
"""

import os
import csv
from typing import Optional, Tuple, Dict, Any

import numpy as np

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DATA_S_PATH = os.path.join(RAW_DATA_DIR, "DATA_S.npz")

FS = 6400.0
F_NOM = 50.0
CYCLE_SAMPLES = int(FS / F_NOM)
MA_WINDOW = 256

BASELINE_SECONDS = 0.25
MIN_BASELINE_WINDOWS = 5

# negative-sequence detector parameters
R2_SIGMA_MULT = 5.0
R2_ABS_MIN = 0.12
MIN_CONSEC_WINDOWS = 2

V_IDX = [0, 1, 2]
I_IDX = [3, 4, 5]

V_STEP_VOLTS = 18.310
I_STEP_AMPS = 4.314

print(f"Loading DATA_S from: {DATA_S_PATH} ...")

DATA_S = np.load(DATA_S_PATH)["DATA_S"]
MAX_SAMPLES = 12053
if DATA_S.shape[0] > MAX_SAMPLES:
    DATA_S = DATA_S[:MAX_SAMPLES]
N_EVENTS, N_CHANNELS, N_SAMPLES = DATA_S.shape
print(f"Dataset initialized with shape: {DATA_S.shape}")


# =============================================================================
# PREPROCESSING
# =============================================================================


def demean_and_detrend(x: np.ndarray, ma_win: int = MA_WINDOW) -> np.ndarray:
    x = x.astype(float)
    x = x - np.mean(x)

    if ma_win > 1:
        kernel = np.ones(ma_win) / ma_win
        trend = np.convolve(x, kernel, mode="same")
        x = x - trend

    return x


def window_signal(event_6ch: np.ndarray, win_size: int = CYCLE_SAMPLES) -> np.ndarray:
    _, n = event_6ch.shape
    n_windows = n // win_size
    trimmed = event_6ch[:, : n_windows * win_size]
    windows = trimmed.reshape(6, n_windows, win_size)
    return np.transpose(windows, (1, 0, 2))


def robust_mean_std(x: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    robust_std = 1.4826 * mad
    return med, robust_std


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def compute_symmetrical_components(
    three_phase_win: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Returns RMS magnitudes of (I0, I1, I2) or (V0, V1, V2) depending on input.
    """
    alpha = np.exp(2j * np.pi / 3)

    T = (1 / 3) * np.array(
        [
            [1, 1, 1],
            [1, alpha**2, alpha],
            [1, alpha, alpha**2],
        ],
        dtype=complex,
    )

    comp012 = T @ three_phase_win
    c0, c1, c2 = comp012

    comp0_rms = np.sqrt(np.mean(np.abs(c0) ** 2))
    comp1_rms = np.sqrt(np.mean(np.abs(c1) ** 2))
    comp2_rms = np.sqrt(np.mean(np.abs(c2) ** 2))

    return comp0_rms, comp1_rms, comp2_rms


def compute_event_features(windows: np.ndarray) -> Dict[str, np.ndarray]:
    n_windows = windows.shape[0]

    I0 = np.zeros(n_windows)
    I1 = np.zeros(n_windows)
    I2 = np.zeros(n_windows)

    for w in range(n_windows):
        win = windows[w]
        I0_w, I1_w, I2_w = compute_symmetrical_components(win[I_IDX])
        I0[w], I1[w], I2[w] = I0_w, I1_w, I2_w

    return {"I0": I0, "I1": I1, "I2": I2}


# =============================================================================
# BASELINE DETECTOR
# =============================================================================


def detect_fault_start_negative_sequence(
    features: Dict[str, np.ndarray],
    fs: float = FS,
    win_size: int = CYCLE_SAMPLES,
    baseline_seconds: float = BASELINE_SECONDS,
    min_baseline_windows: int = MIN_BASELINE_WINDOWS,
    r2_sigma_mult: float = R2_SIGMA_MULT,
    r2_abs_min: float = R2_ABS_MIN,
    min_consec_windows: int = MIN_CONSEC_WINDOWS,
) -> Tuple[Optional[float], np.ndarray, Dict[str, Any]]:
    """
    Negative-sequence baseline detector.

    Score:
        R2 = |I2| / |I1|

    Trigger:
        R2 > r2_abs_min OR z_R2 > r2_sigma_mult
    """
    eps = 1e-12
    I1 = features["I1"]
    I2 = features["I2"]

    n_windows = len(I1)
    n_prefault = max(min_baseline_windows, int(baseline_seconds * fs) // win_size)
    n_prefault = min(n_prefault, n_windows)

    I1_base = I1[:n_prefault]
    I1_floor = max(0.10 * float(np.median(np.abs(I1_base))), 1e-2)
    I1_safe = np.where(np.abs(I1) < I1_floor, I1_floor, I1)

    R2 = I2 / (I1_safe + eps)

    r2_med, r2_std = robust_mean_std(R2[:n_prefault])
    z_R2 = (R2 - r2_med) / (r2_std + eps)

    trigger = (R2 > r2_abs_min) | (z_R2 > r2_sigma_mult)

    run = 0
    start_win = None
    for w in range(n_prefault, n_windows):
        if trigger[w]:
            run += 1
            if run == min_consec_windows:
                start_win = w - min_consec_windows + 1
                break
        else:
            run = 0

    fault_flag = np.zeros(n_windows, dtype=int)

    if start_win is not None:
        fault_flag[start_win:] = 1
        start_time_s = start_win * win_size / fs
        info = {
            "r2_abs_threshold": float(r2_abs_min),
            "r2_z_threshold": float(r2_sigma_mult),
            "r2_at_start": float(R2[start_win]),
            "baseline_median": float(r2_med),
            "baseline_robust_std": float(r2_std),
        }
        return start_time_s, fault_flag, info

    info = {
        "r2_abs_threshold": float(r2_abs_min),
        "r2_z_threshold": float(r2_sigma_mult),
        "r2_at_start": None,
        "baseline_median": float(r2_med),
        "baseline_robust_std": float(r2_std),
    }
    return None, fault_flag, info


# =============================================================================
# EXECUTION
# =============================================================================


def run_fault_detection():
    results = []
    print(f"Executing negative-sequence baseline on {N_EVENTS} events...")

    for evt_idx in range(N_EVENTS):
        raw_event = DATA_S[evt_idx].astype(float)

        raw_event[V_IDX, :] *= V_STEP_VOLTS
        raw_event[I_IDX, :] *= I_STEP_AMPS

        detrended = np.zeros_like(raw_event)
        for ch in range(N_CHANNELS):
            detrended[ch] = demean_and_detrend(raw_event[ch], ma_win=MA_WINDOW)

        windows = window_signal(detrended, win_size=CYCLE_SAMPLES)
        feats = compute_event_features(windows)

        start_time_s, fault_flag, info = detect_fault_start_negative_sequence(feats)

        results.append(
            {
                "event_idx": evt_idx,
                "fault_start_s": start_time_s,
                "r2_abs_threshold": info["r2_abs_threshold"],
                "r2_z_threshold": info["r2_z_threshold"],
                "r2_at_start": info["r2_at_start"],
                "baseline_median": info["baseline_median"],
                "baseline_robust_std": info["baseline_robust_std"],
            }
        )

        if (evt_idx + 1) % 500 == 0:
            print(f"Processed {evt_idx + 1} / {N_EVENTS} events.")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_csv = os.path.join(PROCESSED_DIR, "baseline_negative_sequence_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_idx",
                "fault_start_s",
                "r2_abs_threshold",
                "r2_z_threshold",
                "r2_at_start",
                "baseline_median",
                "baseline_robust_std",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["event_idx"],
                    (
                        round(r["fault_start_s"], 6)
                        if r["fault_start_s"] is not None
                        else ""
                    ),
                    r["r2_abs_threshold"],
                    r["r2_z_threshold"],
                    r["r2_at_start"],
                    r["baseline_median"],
                    r["baseline_robust_std"],
                ]
            )

    print(f"\nProcessing complete. Results persisted to: {output_csv}\n")
    return results


if __name__ == "__main__":
    run_fault_detection()
