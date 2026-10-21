# A Leakage-Free Benchmark of Self-Supervised and Semi-Supervised Representation Learning for IoT Intrusion Detection Under Label Scarcity

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--7016--1833-green.svg)](https://orcid.org/0000-0002-7016-1833)

**Author:** Ramyar A. Teimoor (Corresponding Author)  
**Affiliations:**  
1. Department of Computer, College of Science, University of Sulaimani, Sulaimanyah, Kurdistan Region, Iraq  
2. Information Technology Department, Kurdistan Technical Institute, Sulaimanyah, Kurdistan Region, Iraq  
**Contact:** `ramyar.teimoor@univsul.edu.iq`  
**ORCID:** [0000-0002-7016-1833](https://orcid.org/0000-0002-7016-1833)

---

## 📌 Abstract

Evaluating machine learning for network intrusion detection across Internet of Things (IoT) ecosystems is severely compromised by pervasive duplicate data leakage in standard benchmarks and acute operational label scarcity. In widely cited flow benchmarks such as RT-IoT2022, volumetric attack bursts produce repetitive packet profiles, creating substantial cross-partition duplicate leakage (76.06%) that artificially inflates published detection scores. Concurrently, human security analysts can inspect fewer than 5% of high-velocity telemetry streams.

This repository provides an **exact-duplicate-audited benchmark** comprising 17,710 unique network flows derived from RT-IoT2022 and evaluates thirteen configurations covering eleven distinct ML architectures across unsupervised, supervised, tabular self-supervised (SSL), and semi-supervised paradigms. Using a rigorous twenty-repetition randomized protocol under 1% (123 labeled flows) and 5% (619 labeled flows) label budgets, all models are evaluated on identical deduplicated splits across an independent test set of 3,542 flows.

---

## 🏆 Key Empirical Findings (20-Repetition Suite)

1. **Tree Ensembles Decisively Win Under Label Scarcity:** Non-parametric tree ensembles establish superior performance bounds on tabular flows. At 1% label budget (123 flows):
   - **Random Forest:** 93.50% ± 1.02% Macro-F1 (98.57% ± 0.37% ROC-AUC, 4.46% ± 0.92% FPR)
   - **XGBoost:** 92.07% ± 1.35% Macro-F1 (97.18% ± 1.37% ROC-AUC, 4.19% ± 0.95% FPR)
2. **Tabular SSL Outperforms Hybrid Teacher-Student Architectures:**
   - **VIME:** 91.62% ± 1.31% Macro-F1 (94.82% ± 1.48% ROC-AUC)
   - **SCARF:** 91.15% ± 3.54% Macro-F1 (94.38% ± 1.33% ROC-AUC)
   - **Self-IoT-AD (Hybrid):** 83.83% ± 4.59% Macro-F1 (89.67% ± 3.36% ROC-AUC, 10.34% ± 6.76% FPR)
3. **Confirmation Bias in Tabular Consistency Regularization:**
   - In 20-seed ablation tests, **removing teacher consistency yields a +7.49 percentage point gain** (84.59% ± 4.61% → 92.08% ± 1.33% Macro-F1). Pseudo-label perturbations that succeed in computer vision induce severe boundary blurring across tabular coordinate manifolds.
4. **Pretext Feature Utility Confirmed via Linear Probing:**
   - Linear probing on frozen pretext representations attains **87.59% ± 3.22% Macro-F1** and 4.70% ± 1.86% FPR, significantly outperforming end-to-end semi-supervised fine-tuning ($p = 0.0042$, Cohen's $d_z = -0.76$).
5. **Edge Deployment Qualification:**
   - Dynamic INT8 post-training quantization compresses Self-IoT-AD from 130.9 KB to **48.6 KB** ($2.69\times$ compression) with a host throughput of **255,993 flows/sec** (3.91 $\mu$s/flow).

---

## 📊 Benchmark Summary Table (1% Supervision Budget, $N=20$)

| Model Architecture | Learning Paradigm | Precision (%) | Recall (%) | Positive F1 (%) | Macro-F1 (%) | FPR (%) | ROC-AUC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (1%)** | Supervised Tree | **91.11 ± 1.57** | **91.61 ± 2.99** | **91.32 ± 1.42** | **93.50 ± 1.02** | 4.46 ± 0.92 | **98.57 ± 0.37** |
| **XGBoost (1%)** | Supervised Tree | 91.24 ± 1.81 | 87.50 ± 3.27 | 89.29 ± 1.89 | 92.07 ± 1.35 | 4.19 ± 0.95 | 97.18 ± 1.37 |
| **VIME (1%)** | Tabular SSL | 88.11 ± 2.69 | 89.72 ± 3.05 | 88.85 ± 1.72 | 91.62 ± 1.31 | 6.07 ± 1.71 | 94.82 ± 1.48 |
| **SCARF (1%)** | Tabular SSL | 86.57 ± 6.77 | 90.99 ± 3.52 | 88.48 ± 3.82 | 91.15 ± 3.54 | 7.51 ± 5.81 | 94.38 ± 1.33 |
| **Linear Probe (1%)** | Tabular SSL | 89.30 ± 3.31 | 77.65 ± 7.52 | 82.81 ± 4.84 | 87.59 ± 3.22 | 4.70 ± 1.86 | 94.06 ± 1.49 |
| **Supervised MLP (1%)** | Supervised Deep | 89.83 ± 1.45 | 67.84 ± 9.47 | 76.93 ± 6.56 | 83.84 ± 4.15 | 3.79 ± 0.73 | 92.17 ± 2.26 |
| **Self-IoT-AD (1%)** | SSL + Semi-Supervised | 80.20 ± 9.04 | 77.70 ± 6.43 | 78.44 ± 5.36 | 83.83 ± 4.59 | 10.34 ± 6.76 | 89.67 ± 3.36 |
| **Logistic Regression (1%)** | Supervised Linear | 90.44 ± 1.16 | 66.73 ± 4.58 | 76.68 ± 2.97 | 83.65 ± 1.86 | 3.52 ± 0.59 | 91.81 ± 1.52 |
| **Mean Teacher (1%)** | Semi-Supervised | 78.25 ± 11.61 | 79.93 ± 11.36 | 77.81 ± 6.97 | 82.92 ± 6.30 | 12.96 ± 10.83 | 89.29 ± 5.20 |
| **Deep Autoencoder** | Unsupervised Deep | 82.16 ± 2.03 | 54.17 ± 9.43 | 64.95 ± 7.20 | 75.94 ± 4.51 | 5.72 ± 0.33 | 89.06 ± 2.68 |
| **FixMatch (1%)** | Semi-Supervised | 87.59 ± 11.08 | 53.26 ± 21.71 | 62.02 ± 17.03 | 74.28 ± 9.85 | 5.82 ± 9.77 | 91.21 ± 2.25 |
| **Isolation Forest** | Unsupervised Tree | 88.05 ± 1.44 | 70.31 ± 3.98 | 78.12 ± 2.66 | 84.43 ± 1.73 | 4.74 ± 0.66 | 95.27 ± 0.32 |
| **One-Class SVM** | Unsupervised Kernel | 88.76 ± 3.67 | 12.50 ± 15.15 | 19.18 ± 19.32 | 50.56 ± 10.82 | 0.66 ± 0.69 | 83.99 ± 3.26 |

---

## 📂 Repository Directory Layout

```text
self-iot-ad-benchmark/
├── README.md                          # Project documentation and benchmark overview
├── INSTRUCTIONS_TO_UPLOAD.md          # Step-by-step GitHub upload guide for authors
├── LICENSE                            # MIT License
├── requirements.txt                   # Dependency specifications
├── code/
│   ├── prepare_data.py                # Data ingestion, deduplication, 1-NN leakage audit, stratification
│   ├── model_and_algorithms.py        # PyTorch neural encoder, contrastive pretext head, momentum teacher
│   ├── run_experiments.py             # Main multi-seed benchmark runner (11 architectures, 20 seeds)
│   ├── run_extended_benchmarks.py     # Linear probe suite, unified ablations, INT8 dynamic quantization
│   ├── statistical_analysis.py        # Wilcoxon signed-rank tests, Holm-Bonferroni corrections, d_z
│   └── visualize_results.py           # Publication-grade chart generation (ROC/PR curves, ablations)
├── data/
│   ├── raw/
│   │   └── rt_iot2022_curated.csv     # Raw feature telemetry from RT-IoT2022
│   ├── processed/
│   │   └── rt_iot2022_splits.npz      # Deduplicated, verified splits (17,710 unique flows)
│   └── checksums/
│       └── checksums.txt              # SHA-256 data integrity hashes
├── results/
│   ├── experiment_results_20reps.json # Complete numerical outputs across all 20 repetitions
│   ├── models/
│   │   ├── self_iot_ad_fp32.pt        # Trained FP32 baseline checkpoint
│   │   └── self_iot_ad_int8.pt        # Post-training INT8 quantized checkpoint (48.6 KB)
│   └── raw_predictions/               # Test set prediction vectors per seed
├── tables/                            # Verified experimental CSV tables
└── figures/                           # High-resolution publication figures (PNG & PDF)
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Clone the repository and install required packages:

```bash
git clone https://github.com/ramyarabdulrhaman/self-iot-ad-benchmark.git
cd self-iot-ad-benchmark
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Data Partitions and Leakage Audit

Run the data preparation and 1-NN leakage audit script:

```bash
python code/prepare_data.py
```
*Output:* Validates SHA-256 checksums, computes cross-partition duplicate leakage (confirming 0.45% deduplicated vs. 76.06% raw), and saves `data/processed/rt_iot2022_splits.npz`.

### 3. Run Benchmark Suite

To execute the 20-repetition benchmark across all eleven architectures:

```bash
python code/run_experiments.py --reps 20 --seeds 1000-1893
```

To run the extended multi-seed linear probe and INT8 edge quantization:

```bash
python code/run_extended_benchmarks.py
```

### 4. Compute Statistical Significance & Tables

Generate Holm-Bonferroni adjusted $p$-values and effect sizes:

```bash
python code/statistical_analysis.py
```

### 5. Generate Publication Figures

Render all ROC curves, PR curves, and ablation figures:

```bash
python code/visualize_results.py
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
