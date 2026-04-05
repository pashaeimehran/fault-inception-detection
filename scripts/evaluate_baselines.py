import os
import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MANUAL_CSV = os.path.join(PROCESSED_DIR, "labelstudio_evaluation_flat.csv")
DIDT_CSV = os.path.join(PROCESSED_DIR, "baseline_didt_results.csv")
NEGSEQ_CSV = os.path.join(PROCESSED_DIR, "baseline_negative_sequence_results.csv")

MERGED_OUT = os.path.join(PROCESSED_DIR, "manual_vs_baselines_merged.csv")
SUMMARY_OUT = os.path.join(PROCESSED_DIR, "manual_vs_baselines_summary.csv")
BRANCH_SUMMARY_OUT = os.path.join(
    PROCESSED_DIR, "manual_vs_baselines_branch_summary.csv"
)

# Optional: set to "branch" if you later rename the column
GROUP_COL_CANDIDATES = ["group", "branch"]


# =============================================================================
# HELPERS
# =============================================================================


def normalize_prediction_column(x):
    """Convert empty strings / NaN to np.nan, otherwise float."""
    if pd.isna(x) or x == "":
        return np.nan
    return float(x)


def get_group_column(df: pd.DataFrame) -> str | None:
    for col in GROUP_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def prepare_manual_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Robust normalization
    if "predicted_t0_s" in df.columns:
        df["predicted_t0_s"] = df["predicted_t0_s"].apply(normalize_prediction_column)
    else:
        raise ValueError("Manual CSV must contain 'predicted_t0_s'.")

    if "true_fault_start_s" in df.columns:
        df["true_fault_start_s"] = df["true_fault_start_s"].apply(
            normalize_prediction_column
        )
    else:
        raise ValueError("Manual CSV must contain 'true_fault_start_s'.")

    # Ground truth positive: manually confirmed fault-consistent event
    if "fault_consistent_event" in df.columns:
        df["gt_positive"] = df["fault_consistent_event"].astype(bool)
    elif "has_true_fault_start" in df.columns:
        df["gt_positive"] = df["has_true_fault_start"].astype(bool)
    else:
        raise ValueError(
            "Manual CSV must contain either 'fault_consistent_event' or 'has_true_fault_start'."
        )

    # Fusion prediction is already part of the manual file
    if "has_prediction" in df.columns:
        df["fusion_pred_positive"] = df["has_prediction"].astype(bool)
    else:
        df["fusion_pred_positive"] = df["predicted_t0_s"].notna()

    df["fusion_pred_t0_s"] = df["predicted_t0_s"]

    required = [
        "event_idx",
        "gt_positive",
        "true_fault_start_s",
        "fusion_pred_positive",
        "fusion_pred_t0_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required manual columns: {missing}")

    return df


def prepare_baseline_df(path: str, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    if "event_idx" not in df.columns or "fault_start_s" not in df.columns:
        raise ValueError(
            f"{method_name} CSV must contain columns 'event_idx' and 'fault_start_s'."
        )

    df["fault_start_s"] = df["fault_start_s"].apply(normalize_prediction_column)
    df[f"{method_name}_pred_positive"] = df["fault_start_s"].notna()
    df[f"{method_name}_pred_t0_s"] = df["fault_start_s"]

    keep_cols = [
        "event_idx",
        f"{method_name}_pred_positive",
        f"{method_name}_pred_t0_s",
    ]
    return df[keep_cols]


def compute_metrics(df: pd.DataFrame, pred_positive_col: str, pred_t0_col: str) -> dict:
    """
    Detection metrics:
        TP = predicted positive AND ground-truth positive
        FP = predicted positive AND ground-truth negative
        FN = predicted negative AND ground-truth positive
        TN = predicted negative AND ground-truth negative

    Timing metrics:
        computed only on matched positives = TP
    """
    pred = df[pred_positive_col].astype(bool)
    gt = df["gt_positive"].astype(bool)

    tp_mask = pred & gt
    fp_mask = pred & (~gt)
    fn_mask = (~pred) & gt
    tn_mask = (~pred) & (~gt)

    tp = int(tp_mask.sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())
    tn = int(tn_mask.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else np.nan
    )

    timing_df = df.loc[tp_mask, ["event_idx", "true_fault_start_s", pred_t0_col]].copy()
    timing_df["timing_error_s"] = (
        timing_df[pred_t0_col] - timing_df["true_fault_start_s"]
    )
    timing_df["abs_timing_error_s"] = timing_df["timing_error_s"].abs()

    median_abs_ms = np.nan
    median_signed_ms = np.nan
    within_20ms = np.nan
    within_40ms = np.nan

    if len(timing_df) > 0:
        median_abs_ms = 1000.0 * float(timing_df["abs_timing_error_s"].median())
        median_signed_ms = 1000.0 * float(timing_df["timing_error_s"].median())
        within_20ms = 100.0 * float((timing_df["abs_timing_error_s"] <= 0.020).mean())
        within_40ms = 100.0 * float((timing_df["abs_timing_error_s"] <= 0.040).mean())

    return {
        "method": pred_positive_col.replace("_pred_positive", ""),
        "N": int(len(df)),
        "actual_positives": int(gt.sum()),
        "actual_negatives": int((~gt).sum()),
        "predicted_positives": int(pred.sum()),
        "predicted_negatives": int((~pred).sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision_percent": 100.0 * precision if not np.isnan(precision) else np.nan,
        "recall_percent": 100.0 * recall if not np.isnan(recall) else np.nan,
        "specificity_percent": (
            100.0 * specificity if not np.isnan(specificity) else np.nan
        ),
        "f1_percent": 100.0 * f1 if not np.isnan(f1) else np.nan,
        "matched_disturbances_for_timing": int(len(timing_df)),
        "median_timing_error_ms": median_signed_ms,
        "median_abs_timing_error_ms": median_abs_ms,
        "within_20ms_percent": within_20ms,
        "within_40ms_percent": within_40ms,
    }


def compute_branch_metrics(
    df: pd.DataFrame,
    branch_col: str,
    pred_positive_col: str = "fusion_pred_positive",
    pred_t0_col: str = "fusion_pred_t0_s",
) -> pd.DataFrame:
    """
    Compute fusion performance separately for each manual branch/group.
    """
    rows = []

    for branch_name, subdf in df.groupby(branch_col, dropna=False):
        metrics = compute_metrics(subdf, pred_positive_col, pred_t0_col)

        rows.append(
            {
                "branch": branch_name,
                "N": metrics["N"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TN": metrics["TN"],
                "precision_percent": metrics["precision_percent"],
                "recall_percent": metrics["recall_percent"],
                "specificity_percent": metrics["specificity_percent"],
                "f1_percent": metrics["f1_percent"],
                "matched_disturbances_for_timing": metrics[
                    "matched_disturbances_for_timing"
                ],
                "median_timing_error_ms": metrics["median_timing_error_ms"],
                "median_abs_timing_error_ms": metrics["median_abs_timing_error_ms"],
                "within_20ms_percent": metrics["within_20ms_percent"],
                "within_40ms_percent": metrics["within_40ms_percent"],
            }
        )

    out = pd.DataFrame(rows)

    preferred_order = [
        "fast_path",
        "fusion_high",
        "fusion_mid",
        "fill_nonfast_detected",
        "rejected_random",
        "random_global",
    ]
    order_map = {name: i for i, name in enumerate(preferred_order)}
    out["_order"] = out["branch"].map(lambda x: order_map.get(x, 999))
    out = (
        out.sort_values(["_order", "branch"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    return out


# =============================================================================
# MAIN
# =============================================================================


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    manual_df = prepare_manual_df(MANUAL_CSV)
    didt_df = prepare_baseline_df(DIDT_CSV, "didt")
    negseq_df = prepare_baseline_df(NEGSEQ_CSV, "negseq")

    # Keep only the manually reviewed events and join baseline outputs onto them
    merged = manual_df.merge(didt_df, on="event_idx", how="left")
    merged = merged.merge(negseq_df, on="event_idx", how="left")

    # Fill missing baseline predictions as negative / NaN time
    for method in ["didt", "negseq"]:
        pred_col = f"{method}_pred_positive"
        t0_col = f"{method}_pred_t0_s"

        if pred_col not in merged.columns:
            merged[pred_col] = False
        else:
            merged[pred_col] = merged[pred_col].fillna(False).astype(bool)

        if t0_col not in merged.columns:
            merged[t0_col] = np.nan

    # Compute timing errors per method for inspection
    for method in ["fusion", "didt", "negseq"]:
        pred_col = f"{method}_pred_positive"
        t0_col = f"{method}_pred_t0_s"
        err_col = f"{method}_timing_error_s"
        abs_err_col = f"{method}_abs_timing_error_s"

        merged[err_col] = np.where(
            merged["gt_positive"] & merged[pred_col],
            merged[t0_col] - merged["true_fault_start_s"],
            np.nan,
        )
        merged[abs_err_col] = np.abs(merged[err_col])

    # Save merged per-event table
    merged.to_csv(MERGED_OUT, index=False)

    # Overall summary metrics
    summary_rows = [
        compute_metrics(merged, "fusion_pred_positive", "fusion_pred_t0_s"),
        compute_metrics(merged, "didt_pred_positive", "didt_pred_t0_s"),
        compute_metrics(merged, "negseq_pred_positive", "negseq_pred_t0_s"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    # Branch-wise fusion summary
    group_col = get_group_column(merged)
    if group_col is not None:
        branch_summary_df = compute_branch_metrics(
            merged,
            branch_col=group_col,
            pred_positive_col="fusion_pred_positive",
            pred_t0_col="fusion_pred_t0_s",
        )
        branch_summary_df.to_csv(BRANCH_SUMMARY_OUT, index=False)

        print("\n=== Branch-wise fusion summary ===")
        print(
            branch_summary_df[
                [
                    "branch",
                    "N",
                    "TP",
                    "FP",
                    "precision_percent",
                    "median_timing_error_ms",
                ]
            ].to_string(index=False)
        )
        print(f"\nBranch summary file written to: {BRANCH_SUMMARY_OUT}")
    else:
        print(
            "\nNo 'group' or 'branch' column found in manual CSV; skipping branch-wise summary."
        )

    print("\n=== Overall summary comparison ===")
    print(summary_df.to_string(index=False))
    print(f"\nMerged per-event file written to: {MERGED_OUT}")
    print(f"Summary file written to: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
