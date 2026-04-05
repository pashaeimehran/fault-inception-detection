import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
NPZ_PATH = Path(r"..\digital-fault-recording-database\DATA_S.npz")
SUBSET_CSV = Path("verification_subset_300.csv")
OUTPUT_DIR = Path("verification_data")
OUTPUT_DIR.mkdir(exist_ok=True)

FS = 6400.0  # Hz
V_SCALE = 18.310  # V / LSB
I_SCALE = 4.314  # A / LSB


# ---------------------------------------------------------
# Fortescue symmetrical components
# ---------------------------------------------------------
def compute_symmetrical_components(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """
    Compute sample-wise Fortescue symmetrical components for three real-valued phase signals.

    Returns:
        x0, x1, x2 : complex np.ndarray
            Zero-, positive-, and negative-sequence components.
    """
    alpha = np.exp(1j * 2 * np.pi / 3)

    x0 = (a + b + c) / 3.0
    x1 = (a + alpha * b + alpha**2 * c) / 3.0
    x2 = (a + alpha**2 * b + alpha * c) / 3.0

    return x0, x1, x2


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    if not SUBSET_CSV.exists():
        raise FileNotFoundError(f"Subset CSV not found: {SUBSET_CSV}")

    # -----------------------------------------------------
    # Load subset selection
    # -----------------------------------------------------
    subset = pd.read_csv(SUBSET_CSV)

    if "event_idx" not in subset.columns:
        raise ValueError(
            "verification_subset_300.csv must contain an 'event_idx' column."
        )

    # Keep unique indices, sorted for reproducibility
    event_indices = np.sort(subset["event_idx"].dropna().astype(int).unique())
    print(f"Unique events to extract: {len(event_indices)}")

    # -----------------------------------------------------
    # Load NPZ dataset
    # -----------------------------------------------------
    data = np.load(NPZ_PATH)
    print("Keys in dataset:", data.files)

    if "DATA_S" not in data.files:
        raise KeyError(f"Expected key 'DATA_S' in NPZ file, found: {data.files}")

    data_s = data["DATA_S"]  # shape: (events, 6, samples)
    print(f"Loaded DATA_S with shape: {data_s.shape}")

    if data_s.ndim != 3:
        raise ValueError(
            f"Expected DATA_S to have shape (events, signals, samples), got {data_s.shape}"
        )

    n_events, n_signals, n_samples = data_s.shape
    if n_signals != 6:
        raise ValueError(f"Expected 6 signals per event in DATA_S, got {n_signals}")

    # DATA_S convention:
    # [0,1,2] = phase voltages (Va, Vb, Vc)
    # [3,4,5] = phase currents (Ia, Ib, Ic)
    voltages = np.transpose(data_s[:, 0:3, :], (0, 2, 1))  # -> (events, samples, 3)
    currents = np.transpose(data_s[:, 3:6, :], (0, 2, 1))  # -> (events, samples, 3)

    if currents.shape != voltages.shape:
        raise ValueError(
            f"Currents and voltages must have the same shape, got "
            f"{currents.shape=} and {voltages.shape=}"
        )

    _, _, n_channels = currents.shape
    if n_channels != 3:
        raise ValueError(f"Expected 3 phase channels, got {n_channels}")

    print(f"Processed dataset shape: {currents.shape} = (events, samples, phases)")

    # -----------------------------------------------------
    # Validate indices
    # -----------------------------------------------------
    invalid = event_indices[(event_indices < 0) | (event_indices >= n_events)]
    if len(invalid) > 0:
        raise IndexError(
            f"Found invalid event_idx values outside [0, {n_events - 1}]: {invalid[:10]}"
        )

    # Time vector
    t = np.arange(n_samples, dtype=np.float64) / FS

    # -----------------------------------------------------
    # Extract only selected events in one shot
    # -----------------------------------------------------
    selected_currents = currents[event_indices]  # shape: (n_selected, n_samples, 3)
    selected_voltages = voltages[event_indices]  # shape: (n_selected, n_samples, 3)

    print(f"Loaded selected subset into memory: {selected_currents.shape}")

    # -----------------------------------------------------
    # Per-event export
    # -----------------------------------------------------
    for local_i, event_idx in enumerate(event_indices):
        cur = selected_currents[local_i]
        vol = selected_voltages[local_i]

        Ia = cur[:, 0].astype(np.float64, copy=False) * I_SCALE
        Ib = cur[:, 1].astype(np.float64, copy=False) * I_SCALE
        Ic = cur[:, 2].astype(np.float64, copy=False) * I_SCALE

        Va = vol[:, 0].astype(np.float64, copy=False) * V_SCALE
        Vb = vol[:, 1].astype(np.float64, copy=False) * V_SCALE
        Vc = vol[:, 2].astype(np.float64, copy=False) * V_SCALE

        # Sample-wise symmetrical components
        I0, I1, I2 = compute_symmetrical_components(Ia, Ib, Ic)
        V0, V1, V2 = compute_symmetrical_components(Va, Vb, Vc)

        # predicted onset marker
        row_match = subset.loc[subset["event_idx"] == event_idx]

        pred_t0 = None
        if not row_match.empty and pd.notna(row_match.iloc[0]["fault_start_s"]):
            pred_t0 = float(row_match.iloc[0]["fault_start_s"])

        pred_t0_marker = np.zeros_like(t, dtype=np.float64)

        if pred_t0 is not None:
            idx_t0 = int(np.argmin(np.abs(t - pred_t0)))

            # small spike around predicted t0
            left = max(0, idx_t0 - 3)
            right = min(len(t), idx_t0 + 4)

            pred_t0_marker[left:right] = 1.0

        # Build compact export table
        df_event = pd.DataFrame(
            {
                "t": t,
                "Ia": Ia,
                "Ib": Ib,
                "Ic": Ic,
                "Va": Va,
                "Vb": Vb,
                "Vc": Vc,
                # Current sequence components: real / imag / magnitude
                "I0_real": I0.real,
                "I0_imag": I0.imag,
                "I0_abs": np.abs(I0),
                "I1_real": I1.real,
                "I1_imag": I1.imag,
                "I1_abs": np.abs(I1),
                "I2_real": I2.real,
                "I2_imag": I2.imag,
                "I2_abs": np.abs(I2),
                # Voltage sequence components: real / imag / magnitude
                "V0_real": V0.real,
                "V0_imag": V0.imag,
                "V0_abs": np.abs(V0),
                "V1_real": V1.real,
                "V1_imag": V1.imag,
                "V1_abs": np.abs(V1),
                "V2_real": V2.real,
                "V2_imag": V2.imag,
                "V2_abs": np.abs(V2),
                "pred_t0_marker": pred_t0_marker,
            }
        )

        out_file = OUTPUT_DIR / f"event_{event_idx:05d}.csv"
        df_event.to_csv(out_file, index=False)

    print("Extraction complete.")
    print(f"Files saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
