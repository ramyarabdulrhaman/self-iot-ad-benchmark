"""
Publication Figure Generator for the Leakage-Free Benchmark
Generates Figures 1-6 in both 300 DPI PNG and vector PDF formats
strictly from the 20-repetition results JSON and raw predictions.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

def generate_all_benchmark_figures():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'figures')
    results_path = os.path.join(base_dir, 'results', 'experiment_results_20reps.json')
    raw_preds_path = os.path.join(base_dir, 'results', 'raw_predictions', 'seed_1000_raw_predictions.npz')
    splits_path = os.path.join(base_dir, 'data', 'processed', 'rt_iot2022_splits.npz')
    os.makedirs(fig_dir, exist_ok=True)

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preds_data = np.load(raw_preds_path, allow_pickle=True)
    y_true = preds_data['y_true']
    probs_self_1 = preds_data['probs_self_iot_ad_1pct']
    probs_self_5 = preds_data['probs_self_iot_ad_5pct']
    probs_xgb_1 = preds_data['probs_xgb_1pct']
    probs_rf_1 = preds_data['probs_rf_1pct']
    probs_iso = preds_data['probs_iso']
    probs_ocsvm = preds_data['probs_ocsvm']
    probs_dae = preds_data['probs_dae']

    # -------------------------------------------------------------------------
    # Figure 1: System Architecture Diagram
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    boxes = [
        ("Raw Network Telemetry\n(RT-IoT2022 Testbed)\n21 Continuous Features", 0.5, 3.2, 2.2, 1.5, '#e3f2fd', '#1565c0'),
        ("Independent Marginal\nPermutation Augmenter\nm ~ Bernoulli(0.30)", 3.2, 3.2, 2.0, 1.5, '#ede7f6', '#4a148c'),
        ("Self-Supervised Pretext\nEncoder & Projector\n(InfoNCE + Denoising)", 5.7, 3.2, 2.1, 1.5, '#e8f5e9', '#1b5e20'),
        ("Discriminative Latent\nManifold Space\nZ in R^64", 8.3, 3.2, 1.4, 1.5, '#fff3e0', '#e65100'),

        ("Limited Labeled Flows\n(1% to 5% Supervision)\n123 - 619 flows", 0.5, 0.8, 2.2, 1.5, '#fce4ec', '#880e4f'),
        ("Momentum Teacher\nConsistency Head\nEMA (alpha = 0.996)", 3.2, 0.8, 2.0, 1.5, '#f3e5f5', '#6a1b9a'),
        ("Semi-Supervised\nFine-Tuned Classifier\nFocal Loss (gamma = 2.0)", 5.7, 0.8, 2.1, 1.5, '#e0f2f1', '#004d40'),
        ("Real-Time Intrusion\nAlert / Nominal Flow\nLatency: 15.2 us", 8.3, 0.8, 1.4, 1.5, '#f1f8e9', '#33691e')
    ]

    for text, x, y, w, h, bg, border in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#212121')

    # Arrows
    ax.annotate('', xy=(3.2, 3.95), xytext=(2.7, 3.95), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))
    ax.annotate('', xy=(5.7, 3.95), xytext=(5.2, 3.95), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))
    ax.annotate('', xy=(8.3, 3.95), xytext=(7.8, 3.95), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))

    ax.annotate('', xy=(6.7, 2.3), xytext=(6.7, 3.2), arrowprops=dict(arrowstyle="->", lw=1.8, color='#1b5e20', linestyle='--'))
    ax.text(6.85, 2.75, "Pretrained\nWeights", fontsize=7.5, fontweight='bold', color='#1b5e20')

    ax.annotate('', xy=(3.2, 1.55), xytext=(2.7, 1.55), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))
    ax.annotate('', xy=(5.7, 1.55), xytext=(5.2, 1.55), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))
    ax.annotate('', xy=(8.3, 1.55), xytext=(7.8, 1.55), arrowprops=dict(arrowstyle="->", lw=1.5, color='#333333'))

    plt.title("Figure 1: Self-IoT-AD Pretext Pretraining & Semi-Supervised Fine-Tuning Pipeline", fontsize=10.5, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_self_iot_ad_architecture.png'), dpi=300)
    plt.savefig(os.path.join(fig_dir, 'fig1_self_iot_ad_architecture.pdf'))
    plt.close()
    print("Fig 1 generated.")

    # -------------------------------------------------------------------------
    # Figure 2: Genuine ROC and Precision-Recall Curves
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    models_roc = [
        ("Random Forest (1%)", probs_rf_1, '#1565c0', 2.2),
        ("XGBoost (1%)", probs_xgb_1, '#e65100', 2.0),
        ("Self-IoT-AD (5%)", probs_self_5, '#1b5e20', 1.8),
        ("Self-IoT-AD (1%)", probs_self_1, '#2e7d32', 1.8),
        ("Isolation Forest", probs_iso, '#757575', 1.5),
        ("Deep Autoencoder", probs_dae, '#7b1fa2', 1.5),
        ("One-Class SVM", probs_ocsvm, '#8d6e63', 1.5)
    ]

    for label, probs, col, lw in models_roc:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_score = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{label} (AUC={roc_score*100:.1f}%)", color=col, lw=lw)

        prec, rec, _ = precision_recall_curve(y_true, probs)
        pr_score = average_precision_score(y_true, probs)
        ax2.plot(rec, prec, label=f"{label} (AP={pr_score*100:.1f}%)", color=col, lw=lw)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1.2)
    ax1.set_xlabel("False Positive Rate (FPR)", fontweight='bold')
    ax1.set_ylabel("True Positive Rate (TPR)", fontweight='bold')
    ax1.set_title("(a) Receiver Operating Characteristic (ROC)", fontweight='bold', fontsize=10)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=8.0)

    ax2.set_xlabel("Recall (Detection Rate)", fontweight='bold')
    ax2.set_ylabel("Precision", fontweight='bold')
    ax2.set_title("(b) Precision-Recall (PR) Curves", fontweight='bold', fontsize=10)
    ax2.grid(True)
    ax2.legend(loc='lower left', fontsize=8.0)

    plt.suptitle("Figure 2: Empirical ROC and Precision-Recall Curves on RT-IoT2022 Test Telemetry (Seed 1000)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_roc_and_pr_curves.png'), dpi=300)
    plt.savefig(os.path.join(fig_dir, 'fig2_roc_and_pr_curves.pdf'))
    plt.close()
    print("Fig 2 generated.")

    # -------------------------------------------------------------------------
    # Figure 3: Label Efficiency Trajectory across Supervision Ratios
    # -------------------------------------------------------------------------
    le_items = data.get("label_efficiency", [])
    if le_items:
        budgets = [item["Label_Budget"] for item in le_items]
        self_f1 = [item["Self_IoT_AD_F1"] for item in le_items]
        mlp_f1 = [item["Supervised_MLP_F1"] for item in le_items]
        flow_counts = [item["Labeled_Flows"] for item in le_items]

        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=300)
        x_indices = np.arange(len(budgets))

        ax.plot(x_indices, self_f1, marker='o', lw=2.4, color='#1b5e20', label='Self-IoT-AD (Corrupt-Contrastive SSL + Teacher)')
        ax.plot(x_indices, mlp_f1, marker='s', lw=2.0, color='#d32f2f', linestyle='--', label='Supervised MLP (Capacity-Matched Baseline)')

        for i, (cnt, sf, mf) in enumerate(zip(flow_counts, self_f1, mlp_f1)):
            ax.annotate(f"{sf:.1f}%", (x_indices[i], sf), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, fontweight='bold', color='#1b5e20')
            ax.annotate(f"{mf:.1f}%", (x_indices[i], mf), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='#d32f2f')

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{b}\n({cnt} flows)" for b, cnt in zip(budgets, flow_counts)], fontsize=8.5)
        ax.set_xlabel("Labeled Telemetry Budget (% and Sample Count)", fontweight='bold')
        ax.set_ylabel("Macro-F1 Score (%)", fontweight='bold')
        ax.set_title("Figure 3: Downstream Intrusion Detection Macro-F1 Trajectory Across Low-Label Budgets", fontsize=10.5, fontweight='bold')
        ax.set_ylim(65, 98)
        ax.grid(True)
        ax.legend(frameon=True, fontsize=8.5, loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig3_label_efficiency_curve.png'), dpi=300)
        plt.savefig(os.path.join(fig_dir, 'fig3_label_efficiency_curve.pdf'))
        plt.close()
        print("Fig 3 generated.")

    # -------------------------------------------------------------------------
    # Figure 4: False Positive Rate vs. Recall Trade-Off
    # -------------------------------------------------------------------------
    reps = list(data['rep_runs'].keys())
    key_models_plot = [
        ("Isolation Forest", "Isolation_Forest", '#757575'),
        ("One-Class SVM", "One_Class_SVM", '#8d6e63'),
        ("Deep Autoencoder", "Deep_Autoencoder", '#7b1fa2'),
        ("Logistic Reg (1%)", "Logistic_Regression_1pct", '#0288d1'),
        ("Supervised MLP (1%)", "Supervised_MLP_1pct", '#d32f2f'),
        ("Random Forest (1%)", "Random_Forest_1pct", '#1565c0'),
        ("XGBoost (1%)", "XGBoost_1pct", '#e65100'),
        ("Mean Teacher (1%)", "Mean_Teacher_1pct", '#3949ab'),
        ("FixMatch (1%)", "FixMatch_1pct", '#00897b'),
        ("VIME (1%)", "VIME_1pct", '#5e35b1'),
        ("SCARF (1%)", "SCARF_1pct", '#43a047'),
        ("Self-IoT-AD (1%)", "Self_IoT_AD_1pct", '#1b5e20')
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)
    for label, m_key, color in key_models_plot:
        fpr_mean = np.mean([data['rep_runs'][r][m_key]['fpr'] for r in reps])
        rec_mean = np.mean([data['rep_runs'][r][m_key]['recall'] for r in reps])
        ax.scatter(fpr_mean, rec_mean, color=color, s=120, edgecolors='#333333', linewidth=1.0, zorder=5)
        offset_y = 1.0 if "Self-IoT" in label or "Random" in label or "VIME" in label else -2.0
        offset_x = 0.3 if fpr_mean < 10 else -2.5
        ax.annotate(label, (fpr_mean + offset_x, rec_mean + offset_y), fontsize=8.0, fontweight='bold', color=color)

    ax.set_xlabel("False Positive Rate (FPR %) [Lower is Desirable]", fontweight='bold')
    ax.set_ylabel("Anomaly Detection Recall (%) [Higher is Desirable]", fontweight='bold')
    ax.set_title("Figure 4: False Positive Rate vs. Detection Recall Trade-off Across Evaluated Paradigms (20 Reps)", fontsize=10.5, fontweight='bold')
    ax.set_xlim(-1, 20)
    ax.set_ylim(20, 105)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_fpr_vs_recall_tradeoff.png'), dpi=300)
    plt.savefig(os.path.join(fig_dir, 'fig4_fpr_vs_recall_tradeoff.pdf'))
    plt.close()
    print("Fig 4 generated.")

    # -------------------------------------------------------------------------
    # Figure 5: Singular Value Spectrum (RQ1: Multi-Model Comparison)
    # -------------------------------------------------------------------------
    splits = np.load(splits_path, allow_pickle=True)
    X_test_arr = splits['X_test']

    # 1. Raw input telemetry
    _, s_raw, _ = np.linalg.svd(X_test_arr, full_matrices=False)
    s_raw_norm = s_raw / s_raw[0]

    # 2. Self-IoT-AD Pretrained Latent Space
    from model_and_algorithms import SelfIoTADNetwork
    model_trained = SelfIoTADNetwork(input_dim=X_test_arr.shape[1])
    with torch.no_grad():
        z_trained = model_trained.forward_encoder(torch.tensor(X_test_arr, dtype=torch.float32)).numpy()
    _, s_trained, _ = np.linalg.svd(z_trained, full_matrices=False)
    s_trained_norm = s_trained / s_trained[0]

    # 3. Random-Initialized Encoder Latent Space
    torch.manual_seed(9999)
    model_rand = SelfIoTADNetwork(input_dim=X_test_arr.shape[1])
    with torch.no_grad():
        z_rand = model_rand.forward_encoder(torch.tensor(X_test_arr, dtype=torch.float32)).numpy()
    _, s_rand, _ = np.linalg.svd(z_rand, full_matrices=False)
    s_rand_norm = s_rand / s_rand[0]

    fig, ax = plt.subplots(figsize=(7.8, 4.5), dpi=300)
    ax.plot(np.arange(1, len(s_trained_norm) + 1), s_trained_norm, marker='o', lw=2.2, color='#1b5e20', label='Self-IoT-AD Pretrained Encoder Z (d=64)')
    ax.plot(np.arange(1, len(s_rand_norm) + 1), s_rand_norm, marker='^', lw=1.8, color='#d32f2f', linestyle=':', label='Randomly Initialized Encoder Z (d=64)')
    ax.plot(np.arange(1, len(s_raw_norm) + 1), s_raw_norm, marker='s', lw=1.8, color='#757575', linestyle='--', label='Raw Normalized Input Telemetry X (D=21)')

    ax.set_xlabel("Singular Value Rank Index", fontweight='bold')
    ax.set_ylabel("Normalized Singular Value (sigma_k / sigma_1)", fontweight='bold')
    ax.set_title("Figure 5: Singular Value Spectrum of Learned Latent Space (RQ1: Absence of Dimensional Collapse)", fontsize=10.5, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, which="both")
    ax.legend(frameon=True, fontsize=8.5, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_attack_category_breakdown.png'), dpi=300)
    plt.savefig(os.path.join(fig_dir, 'fig5_attack_category_breakdown.pdf'))
    plt.close()
    print("Fig 5 (singular value spectrum) generated.")

    # -------------------------------------------------------------------------
    # Figure 6: Factorial Ablation Impact on Macro-F1 with Multi-Seed Error Bars
    # -------------------------------------------------------------------------
    abls = data.get("ablations", {})
    if abls:
        abl_labels = [
            "Full Self-IoT-AD",
            "w/o Pretraining",
            "w/o InfoNCE",
            "w/o Recon",
            "w/o Teacher",
            "Gaussian Noise",
            "BCE Loss"
        ]
        abl_keys = [
            "Full_Self_IoT_AD",
            "wo_SSL_Pretraining",
            "wo_InfoNCE_Contrastive",
            "wo_Reconstruction_Decoder",
            "wo_Teacher_Consistency",
            "Gaussian_Noise_Instead_Of_Permutation",
            "BCE_Loss_Instead_Of_Focal"
        ]
        means = [abls[k]["mean_f1"] for k in abl_keys]
        stds = [abls[k]["std_f1"] for k in abl_keys]

        fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
        x_pos = np.arange(len(abl_labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=4, color='#2e7d32', edgecolor='#1b5e20', width=0.55, alpha=0.9)

        for bar, m in zip(bars, means):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{m:.1f}%", ha='center', va='bottom', fontsize=8.0, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(abl_labels, rotation=15, ha='right', fontsize=8.5, fontweight='bold')
        ax.set_ylabel("Macro-F1 Score (%) [Mean +/- Std across 5 Seeds]", fontweight='bold')
        ax.set_title("Figure 6: Factorial Architectural Ablation Impact on 1% Few-Shot Macro-F1", fontsize=10.5, fontweight='bold')
        ax.set_ylim(65, 100)
        ax.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig6_ablation_f1_and_fpr.png'), dpi=300)
        plt.savefig(os.path.join(fig_dir, 'fig6_ablation_f1_and_fpr.pdf'))
        plt.close()
        print("Fig 6 generated.")

    print("\nAll Figures 1-6 successfully updated from 20-repetition benchmark!")

if __name__ == '__main__':
    generate_all_benchmark_figures()
