"""
Overview
--------
Baseline reference implementation using a simple current-derivative (dI/dt)
threshold detector for fault inception estimation in the RTE DFR dataset.

This baseline is intentionally simple and interpretable:
1. Convert raw ADC counts to physical units.
2. Demean and detrend each current channel.
3. Compute the instantaneous derivative of the three phase currents.
4. Use a robust pre-fault baseline to set a detection threshold.
5. Declare fault inception at the first sample where the derivative score
   exceeds threshold for a minimum persistence duration.

This baseline is suitable as a lightweight comparator against the proposed
physics-guided fusion framework.
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
MIN_BASELINE_SAMPLES = int(BASELINE_SECONDS * FS)

# dI/dt detector parameters
DIDT_SIGMA_MULT = 8.0  # threshold = median + k * robust_std
MIN_CONSEC_SAMPLES = 4  # persistence at sample level
SEARCH_MARGIN_SECONDS = 0.06  # allow search a bit beyond baseline

V_IDX = [0, 1, 2]
I_IDX = [3, 4, 5]

V_STEP_VOLTS = 18.310
I_STEP_AMPS = 4.314

print(f"Loading DATA_S from: {DATA_S_PATH} ...")
if os.path.exists(DATA_S_PATH):
    DATA_S = np.load(DATA_S_PATH)["DATA_S"]
    MAX_SAMPLES = 12053
    if DATA_S.shape[0] > MAX_SAMPLES:
        DATA_S = DATA_S[:MAX_SAMPLES]
    N_EVENTS, N_CHANNELS, N_SAMPLES = DATA_S.shape
    print(f"Dataset initialized with shape: {DATA_S.shape}")
else:
    print(f"ERROR: Resource not found at {DATA_S_PATH}")
    DATA_S = np.zeros((1, 6, 1000))
    N_EVENTS, N_CHANNELS, N_SAMPLES = DATA_S.shape


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


def robust_mean_std(x: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    robust_std = 1.4826 * mad
    return med, robust_std


# =============================================================================
# BASELINE DETECTOR
# =============================================================================


def detect_fault_start_didt(
    raw_event: np.ndarray,
    fs: float = FS,
    baseline_seconds: float = BASELINE_SECONDS,
    didt_sigma_mult: float = DIDT_SIGMA_MULT,
    min_consec_samples: int = MIN_CONSEC_SAMPLES,
) -> Tuple[Optional[float], np.ndarray, Dict[str, Any]]:
    """
    Simple dI/dt threshold detector.

    Score:
        max(|dIa/dt|, |dIb/dt|, |dIc/dt|)

    Threshold:
        median(prefault_score) + k * robust_std(prefault_score)

    Returns:
        (fault_start_s, fault_flag, info)
    """
    eps = 1e-12

    Ia, Ib, Ic = raw_event[I_IDX[0]], raw_event[I_IDX[1]], raw_event[I_IDX[2]]

    # instantaneous derivatives
    dIa = np.diff(Ia, prepend=Ia[0]) * fs
    dIb = np.diff(Ib, prepend=Ib[0]) * fs
    dIc = np.diff(Ic, prepend=Ic[0]) * fs

    score = np.maximum.reduce([np.abs(dIa), np.abs(dIb), np.abs(dIc)])

    n_baseline = min(max(int(baseline_seconds * fs), MIN_BASELINE_SAMPLES), len(score))
    baseline = score[:n_baseline]

    score_med, score_std = robust_mean_std(baseline)
    threshold = score_med + didt_sigma_mult * score_std

    # optional absolute floor from baseline max to avoid trivial triggers
    threshold = max(threshold, 2.0 * float(np.max(baseline)) + eps)

    trigger = score > threshold

    run = 0
    start_idx = None
    for k in range(n_baseline, len(trigger)):
        if trigger[k]:
            run += 1
            if run == min_consec_samples:
                start_idx = k - min_consec_samples + 1
                break
        else:
            run = 0

    fault_flag = np.zeros(len(score), dtype=int)

    if start_idx is not None:
        fault_flag[start_idx:] = 1
        start_time_s = start_idx / fs
        info = {
            "threshold": float(threshold),
            "score_at_start": float(score[start_idx]),
            "baseline_median": float(score_med),
            "baseline_robust_std": float(score_std),
        }
        return start_time_s, fault_flag, info

    info = {
        "threshold": float(threshold),
        "score_at_start": None,
        "baseline_median": float(score_med),
        "baseline_robust_std": float(score_std),
    }
    return None, fault_flag, info


# =============================================================================
# EXECUTION
# =============================================================================


def run_fault_detection():
    results = []
    print(f"Executing dI/dt baseline on {N_EVENTS} events...")

    for evt_idx in range(N_EVENTS):
        raw_event = DATA_S[evt_idx].astype(float)

        raw_event[V_IDX, :] *= V_STEP_VOLTS
        raw_event[I_IDX, :] *= I_STEP_AMPS

        detrended = np.zeros_like(raw_event)
        for ch in range(N_CHANNELS):
            detrended[ch] = demean_and_detrend(raw_event[ch], ma_win=MA_WINDOW)

        start_time_s, fault_flag, info = detect_fault_start_didt(detrended)

        results.append(
            {
                "event_idx": evt_idx,
                "fault_start_s": start_time_s,
                "didt_threshold": info["threshold"],
                "didt_score_at_start": info["score_at_start"],
                "baseline_median": info["baseline_median"],
                "baseline_robust_std": info["baseline_robust_std"],
            }
        )

        if (evt_idx + 1) % 500 == 0:
            print(f"Processed {evt_idx + 1} / {N_EVENTS} events.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    output_csv = os.path.join(PROCESSED_DIR, "baseline_didt_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_idx",
                "fault_start_s",
                "didt_threshold",
                "didt_score_at_start",
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
                    r["didt_threshold"],
                    r["didt_score_at_start"],
                    r["baseline_median"],
                    r["baseline_robust_std"],
                ]
            )

    print(f"\nProcessing complete. Results persisted to: {output_csv}\n")
    return results


if __name__ == "__main__":
    run_fault_detection()
