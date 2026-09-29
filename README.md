<div align="center">

# Fault Inception Detection in Real-World Disturbance Data

[![ISGT Europe 2026](https://img.shields.io/badge/ISGT%20Europe-2026-1f6feb.svg)](https://ieee-isgt-europe.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.23111-b31b1b.svg)](https://arxiv.org/abs/2606.23111)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-2da44e.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab.svg)](https://www.python.org/)

**Training-free, physics-guided estimation of fault inception timestamps (t₀) in Digital Fault Recorder (DFR) recordings.**

Reference implementation accompanying our paper accepted at IEEE PES ISGT Europe 2026.

[📄 Read the paper](https://arxiv.org/abs/2606.23111) · [Cite this work](#citation)

</div>

> **At a glance:** The detector analyzes voltage and current waveforms to identify fault-consistent disturbances and estimate their onset time. It uses physical indicators and robust statistics without fitting a model to labeled training data. We applied it to **12,053 RTE recordings**. On a **deliberately sampled, manually reviewed set of 300 events**, it achieved **96.6% recall**, **79.2% precision**, and a **4.2 ms median absolute timing error** for true positives. These evaluation metrics describe the reviewed set, not an estimated prevalence or performance rate across all 12,053 recordings.

> **Paper status:** Accepted at the 2026 IEEE PES Innovative Smart Grid Technologies Europe conference, October 19–22, 2026, Budapest, Hungary. The linked arXiv version is a preprint; proceedings details will be added when available.

---

## How it works

The detector combines interpretable indicators in four stages:

1. **Preprocessing:** Convert ADC values to physical units, remove baseline trends, and form cycle-synchronous windows at 6,400 Hz and 50 Hz (128 samples per cycle).
2. **Feature extraction:** Compute quantities including RMS, symmetrical components (I₀/I₁/I₂), harmonic and high-frequency energy, wavelet features, and current derivatives.
3. **Robust baseline modeling:** Compare features with a pre-event baseline using median- and MAD-based scores.
4. **Hybrid detection:** Combine a fast transient path with cycle-synchronous triggers, persistence checks, and veto logic for non-fault transients.

For each recording, the output includes an estimated inception time `t₀`, when detected, and flags showing which indicators contributed. See the [paper](https://arxiv.org/abs/2606.23111) for the full method and evaluation.

---

## Repository structure

```text
fusion_fault_inception_detection/        # installable Python package
  robust_fault_inception_detector.py      # main detector
  baseline_didt.py                         # dI/dt baseline
  baseline_negative_sequence.py            # negative-sequence baseline
scripts/
  evaluate_baselines.py                    # comparison with manual labels
  export_summarized_fault_debug_events.py  # figure export
evaluation/manual_validation/             # subset and annotation scripts
data/
  raw/                                     # place DATA_S.npz here
  interim/                                 # subset and annotation artifacts
  processed/                               # detector and evaluation results
figures/                                   # example and paper figures
```

---

## Installation

Requires **Python 3.10+**. Clone the repository and install the package:

```bash
git clone https://github.com/pashaeimehran/fault-inception-detection.git
cd fault-inception-detection
pip install -e .
```

This installs the runtime dependencies (`numpy`, `pandas`, and `PyWavelets`). Optional dependencies are available for figure generation and the annotation workflow:

```bash
pip install -e ".[dev]"
pip install -e ".[labelstudio]"
```

Some committed CSV and figure artifacts use **Git LFS**. If you want to inspect those artifacts locally, install Git LFS and run:

```bash
git lfs pull
```

Without the LFS download, those files may contain pointers rather than the underlying data.

### Dataset

The RTE Digital Fault Recording Database is **not bundled** with this repository. Obtain it from the [RTE dataset repository](https://github.com/rte-france/digital-fault-recording-database) and place `DATA_S.npz` at:

```text
data/raw/DATA_S.npz
```

The detector's parameters and physical conversions are configured for these recordings, including their 90 kV setting and 6,400 Hz sampling rate. Applying it to other grids or recorder formats requires checking those assumptions.

---

## Quickstart

From the repository root, run:

```bash
python -m fusion_fault_inception_detection.robust_fault_inception_detector
```

The script reads `data/raw/DATA_S.npz` and writes per-event results to:

```text
data/processed/robust_fault_inception_results.csv
```

The output includes `event_idx`, `fault_start_s` (the estimated `t₀`), `fusion_score_at_start`, feature flags such as `flag_negseq`, `flag_zeroseq`, `flag_thd`, and `flag_wavelet`, plus fast-path and veto indicators. A missing `fault_start_s` means that the detector did not return a fault inception for that recording.

---

## Baselines

Run the two reference baselines from the repository root:

```bash
python -m fusion_fault_inception_detection.baseline_didt
python -m fusion_fault_inception_detection.baseline_negative_sequence
```

They write:

```text
data/processed/baseline_didt_results.csv
data/processed/baseline_negative_sequence_results.csv
```

---

## Evaluation and manual review

The reported metrics were computed on **300 manually reviewed events** selected from the 12,053 recordings. The review set deliberately includes events from different detector outcomes and score groups. It should **not** be treated as a simple random sample of the full recording archive.

The repository includes the detector and baseline outputs, evaluation artifacts, and scripts used to prepare the manual review. After retrieving the LFS files, the baseline comparison can be run with:

```bash
python scripts/evaluate_baselines.py
```

The scripts in `evaluation/manual_validation/` document the subset selection, waveform export, and Label Studio task preparation. **They are research workflow scripts, not yet a one-command reproduction pipeline:** some paths refer to the original local setup and must be adjusted before running them on a fresh clone. Manual annotation also requires Label Studio setup and human review.

To regenerate the detector outputs from the RTE data, run the quickstart and baseline commands above. To repeat the manual review itself, inspect and configure the scripts in `evaluation/manual_validation/` before running them.

The figure export script is available at:

```bash
python scripts/export_summarized_fault_debug_events.py
```

It uses the required local data and outputs to generate figures under `figures/paper/`.

---

## Results

Performance on the **deliberately sampled, manually reviewed 300-event set**:

| Metric | Result |
| --- | ---: |
| Recall | 96.6% |
| Precision | 79.2% |
| Specificity | 76.0% |
| F1-score | 87.0% |
| Median absolute timing error | 4.2 ms |
| Detections within ±20 ms | 70.2% |
| Detections within ±40 ms | 98.3% |

The timing metrics are calculated over true-positive detections. The precision, recall, specificity, and F1-score reflect the composition of the reviewed set; they should not be extrapolated to the entire archive without accounting for the sampling design.

![Representative fault event](figures/event_example.png)

*Representative detection showing phase currents and fusion-score contributions. The red dashed line marks the estimated inception time.*

---

## Citation

If you use this software, the generated annotations, or the method, please cite the accompanying paper:

```bibtex
@inproceedings{oelhaf2026fault,
  author    = {Oelhaf, Julian and Pashaei, Mehran and P{\'e}rez-Toro, Paula Andrea and Kordowich, Georg and Bergler, Christian and Maier, Andreas and J{\"a}ger, Johann and Bayer, Siming},
  title     = {Fault Inception Detection in Real-World Disturbance Data for Power System Protection},
  booktitle = {2026 IEEE PES Innovative Smart Grid Technologies Europe (ISGT Europe)},
  address   = {Budapest, Hungary},
  month     = oct,
  year      = {2026},
  publisher = {IEEE},
  note      = {Accepted for publication. Preprint: arXiv:2606.23111},
}
```

The machine-readable [`CITATION.cff`](CITATION.cff) also provides a citation through GitHub's **Cite this repository** menu. DOI and page numbers will be added once the proceedings are published.

**Authors:** Julian Oelhaf¹,†,* · Mehran Pashaei¹,† · Paula Andrea Pérez-Toro¹ · Georg Kordowich² · Christian Bergler³ · Andreas Maier¹ · Johann Jäger² · Siming Bayer¹  
<sub>¹ Pattern Recognition Lab, FAU Erlangen-Nürnberg · ² Institute of Electrical Energy Systems, FAU Erlangen-Nürnberg · ³ OTH Amberg-Weiden · † Equal contribution · * [julian.oelhaf@fau.de](mailto:julian
