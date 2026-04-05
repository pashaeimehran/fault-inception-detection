import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_CSV = "digital-fault-recording-database\\robust_fault_inception_results.csv"
OUTPUT_CSV = "verification_subset_300.csv"
RANDOM_SEED = 42
TARGET_TOTAL = 300

# Initial target sizes (the script will adapt if some pools are too small)
N_FAST_PATH = 80
N_FUSION_HIGH = 40
N_FUSION_MID = 40
N_REJECTED = 100
N_RANDOM_GLOBAL = 40

FLAG_COLS = [
    "flag_unbalance",
    "flag_negseq",
    "flag_zeroseq",
    "flag_current",
    "flag_voltage",
    "flag_fft",
    "flag_thd",
    "flag_wavelet",
    "flag_drms",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_sample(frame: pd.DataFrame, n: int, name: str) -> pd.DataFrame:
    """Sample up to n rows without crashing."""
    k = min(n, len(frame))
    print(f"{name}: available={len(frame)}, sampled={k}")
    if k == 0:
        return frame.iloc[0:0].copy()
    return frame.sample(n=k, random_state=RANDOM_SEED).copy()


def mark_group(frame: pd.DataFrame, group_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["verification_group"] = group_name
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(INPUT_CSV)
    print("Total samples:", len(df))

    required_cols = [
        "event_idx",
        "fault_start_s",
        "fusion_score_at_start",
        "fast_path_activated",
        *FLAG_COLS,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    # Normalize columns
    for col in ["fast_path_activated", *FLAG_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["fusion_score_at_start"] = pd.to_numeric(
        df["fusion_score_at_start"], errors="coerce"
    )

    detected = df[df["fault_start_s"].notna()].copy()
    rejected = df[df["fault_start_s"].isna()].copy()

    print("Detected events:", len(detected))
    print("Rejected events:", len(rejected))

    # -----------------------------------------------------------------
    # Primary strata
    # -----------------------------------------------------------------
    fast_path = mark_group(
        safe_sample(
            detected[detected["fast_path_activated"] == 1],
            N_FAST_PATH,
            "fast_path",
        ),
        "fast_path",
    )

    fusion_high = mark_group(
        safe_sample(
            detected[
                (detected["fast_path_activated"] == 0)
                & (detected["fusion_score_at_start"] >= 5)
            ],
            N_FUSION_HIGH,
            "fusion_high",
        ),
        "fusion_high",
    )

    fusion_mid = mark_group(
        safe_sample(
            detected[
                (detected["fast_path_activated"] == 0)
                & (detected["fusion_score_at_start"] >= 3)
                & (detected["fusion_score_at_start"] < 5)
            ],
            N_FUSION_MID,
            "fusion_mid",
        ),
        "fusion_mid",
    )

    rejected_random = mark_group(
        safe_sample(
            rejected,
            N_REJECTED,
            "rejected_random",
        ),
        "rejected_random",
    )

    # Build initial selected set
    selected = pd.concat(
        [fast_path, fusion_high, fusion_mid, rejected_random],
        ignore_index=True,
    ).drop_duplicates(subset=["event_idx"])

    # -----------------------------------------------------------------
    # Random global sanity sample from remaining pool
    # -----------------------------------------------------------------
    selected_ids = set(selected["event_idx"].tolist())
    remaining_all = df[~df["event_idx"].isin(selected_ids)].copy()

    random_global = mark_group(
        safe_sample(
            remaining_all,
            N_RANDOM_GLOBAL,
            "random_global",
        ),
        "random_global",
    )

    selected = pd.concat([selected, random_global], ignore_index=True)
    selected = selected.drop_duplicates(subset=["event_idx"]).copy()

    # -----------------------------------------------------------------
    # Adaptive fill to reach TARGET_TOTAL
    # Priority:
    #   1) remaining non-fast-path detections
    #   2) remaining rejected events
    #   3) any remaining events
    # -----------------------------------------------------------------
    deficit = TARGET_TOTAL - len(selected)
    print("Initial subset size:", len(selected))
    print("Deficit to fill:", deficit)

    if deficit > 0:
        selected_ids = set(selected["event_idx"].tolist())

        remaining_nonfast_detected = detected[
            (~detected["event_idx"].isin(selected_ids))
            & (detected["fast_path_activated"] == 0)
        ].copy()

        fill_nonfast = mark_group(
            safe_sample(
                remaining_nonfast_detected,
                deficit,
                "fill_nonfast_detected",
            ),
            "fill_nonfast_detected",
        )

        selected = pd.concat([selected, fill_nonfast], ignore_index=True)
        selected = selected.drop_duplicates(subset=["event_idx"]).copy()

    deficit = TARGET_TOTAL - len(selected)
    print("Deficit after non-fast-path fill:", deficit)

    if deficit > 0:
        selected_ids = set(selected["event_idx"].tolist())

        remaining_rejected = rejected[
            ~rejected["event_idx"].isin(selected_ids)
        ].copy()

        fill_rejected = mark_group(
            safe_sample(
                remaining_rejected,
                deficit,
                "fill_rejected",
            ),
            "fill_rejected",
        )

        selected = pd.concat([selected, fill_rejected], ignore_index=True)
        selected = selected.drop_duplicates(subset=["event_idx"]).copy()

    deficit = TARGET_TOTAL - len(selected)
    print("Deficit after rejected fill:", deficit)

    if deficit > 0:
        selected_ids = set(selected["event_idx"].tolist())

        remaining_all = df[
            ~df["event_idx"].isin(selected_ids)
        ].copy()

        fill_any = mark_group(
            safe_sample(
                remaining_all,
                deficit,
                "fill_any",
            ),
            "fill_any",
        )

        selected = pd.concat([selected, fill_any], ignore_index=True)
        selected = selected.drop_duplicates(subset=["event_idx"]).copy()

    # -----------------------------------------------------------------
    # Final helper columns
    # -----------------------------------------------------------------
    selected["is_detected"] = selected["fault_start_s"].notna().astype(int)
    selected["active_flag_count"] = selected[FLAG_COLS].sum(axis=1)

    # Manual verification columns
    selected["annotator_judgment"] = ""
    selected["annotator_t0_s"] = np.nan
    selected["timing_error_s"] = np.nan
    selected["comments"] = ""

    # Order columns
    front_cols = [
        "event_idx",
        "verification_group",
        "is_detected",
        "fault_start_s",
        "fusion_score_at_start",
        "fast_path_activated",
        "active_flag_count",
        *FLAG_COLS,
        "annotator_judgment",
        "annotator_t0_s",
        "timing_error_s",
        "comments",
    ]
    remaining_cols = [c for c in selected.columns if c not in front_cols]
    selected = selected[front_cols + remaining_cols]

    selected = selected.sort_values(
        by=["verification_group", "event_idx"],
        ascending=[True, True],
    ).reset_index(drop=True)

    selected.to_csv(OUTPUT_CSV, index=False)

    print("\nFinal unique subset size:", len(selected))
    print("\nGroup counts:")
    print(selected["verification_group"].value_counts(dropna=False))
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()