# Fault Inception Detection in Real-World Disturbance Data

This repository contains the **official Python reference implementation** accompanying the paper:

> **Fault Inception Detection in Real-World Disturbance Data for Power System Protection**

It provides a **training-free, physics-guided method** for estimating **fault inception timestamps (t₀)** in large Digital Fault Recorder (DFR) datasets.

The framework is designed for **real transmission-system recordings**, where faults must be distinguished from switching, energization, resonance, and measurement artifacts.

---

## 📖 Overview

Large DFR archives enable protection analysis, relay benchmarking, and post-event investigation, but typically **lack precise onset annotations**.

Recorded waveforms often include:

- switching operations  
- transformer energization  
- resonance phenomena  
- noise and measurement artifacts  

These can **mimic or obscure fault signatures**, complicating reliable onset detection.

This framework addresses the problem without supervised learning by combining:

- **interpretable protection-domain features**
- **robust statistical normalization**
- **rule-based fusion logic**

---

## ⚙️ Method Summary

The detector consists of four stages:

### 1. Preprocessing

- Demeaning and detrending  
- Cycle-synchronous segmentation  
  - 6400 Hz sampling  
  - 50 Hz nominal frequency  
  - 128 samples per cycle  

---

### 2. Feature Extraction

Cycle-level features include:

- **Magnitude**: RMS current/voltage, current change  
- **Sequence components**: zero/negative sequence, imbalance ratios  
- **Spectral/transient**: THD, high-frequency energy, wavelets, dI/dt  

---

### 3. Robust Baseline Modeling

Baseline statistics are estimated from a pre-event region using:

- median  
- MAD  

Modified Z-scores prevent **statistical masking** caused by large disturbances.

---

### 4. Hybrid Detection

Two complementary paths:

**Fast transient path**

- Zero-sequence residuals  
- Current derivatives  
→ sub-cycle response  

**Cycle-synchronous fusion**

- Weighted combination of feature triggers  
- Persistence constraints  
- Veto logic for non-fault disturbances  

Output: estimated onset time **t₀**

---

## 🗃️ Dataset Configuration

Configured for the **RTE Digital Fault Recording Database**:

| Property | Value |
|--------|--------|
| Voltage level | 90 kV |
| Events | 12,053 |
| Channels | 3V + 3I |
| Sampling rate | 6400 Hz |
| Frequency | 50 Hz |
| Samples/cycle | 128 |
| Duration | ~3.28 s |

Each event includes pre-event, disturbance, and post-event behavior.

---

## 📦 Setup

1. Download dataset:  
   <https://github.com/rte-france/digital-fault-recording-database>  

2. Place the dataset file at:

    data/raw/DATA_S.npz

---

## ⚙️ Installation

```bash
pip install numpy matplotlib
```

Optional:

```bash
pip install PyWavelets
```

---

## 🚀 Usage

Run the detector from the repository root:

```bash
python fusion_fault_inception_detection/robust_fault_inception_detector.py
```

Alternatively (module execution):

```bash
python -m fusion_fault_inception_detection.robust_fault_inception_detector
```

---

## 📊 Output

Generates:

```text
robust_fault_inception_results.csv
```

Includes:

- estimated onset time (t₀)
- fusion score
- trigger flags
- veto indicators

---

## 🔬 Reproducing Evaluation

Scripts for manual validation:

```bash
python robust_fault_inception_detector.py
python select_subset_of_samples.py
python extract_verification_events.py
python generate_labelstudio_tasks.py
```

Reproduces the **manual review pipeline (300 events)** used in the paper.

---

## 📊 Performance

Evaluated on **12,053 recordings** with **manual review of 300 events**:

| Metric       | Result |
| ------------ | ------ |
| Recall       | 96.6%  |
| Precision    | 79.2%  |
| Specificity  | 76.0%  |
| F1-score     | 87.0%  |
| Timing error | 4.2 ms |

Median absolute timing error over true positives.

---

## 📜 Citation

```bibtex
@article{oelhaf_2026_fault,
  author = {Oelhaf, J. and Pashaei, M. and Pérez-Toro, P. A. and Kordowich, G. and Bergler, C. and Maier, A. and Jäger, J. and Bayer, S.},
  title = {Fault Inception Detection in Real-World Disturbance Data for Power System Protection},
  journal = {Submitted for publication},
  year = {2026}
}
```

---

## License

BSD-3-Clause License.

---

## 🔍 Example Detection

![Representative fault event](figures/event_example.png)

*Example of a detected fault inception. The top plot shows phase currents, and the bottom plot shows the fusion score contributions. The red dashed line marks the estimated fault inception time.*
