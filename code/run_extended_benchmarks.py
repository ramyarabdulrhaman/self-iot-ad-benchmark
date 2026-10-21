"""
Extended Benchmark Experiments for IoT Self-Supervised Anomaly Detection:
1. 20-Repetition Linear Probe Benchmark (Seeds 1000-1893)
   - Evaluates Linear Probe on frozen encoder representation Z across all 20 seeds
   - Integrates per-repetition Linear Probe metrics into results/experiment_results_20reps.json
2. 20-Repetition Factorial Ablation Benchmark (Seeds 1000-1893)
   - Re-runs all 7 ablation configurations across the full 20-seed suite for unified statistical protocol
3. Edge Device & Quantization Profiling (FP32 vs INT8)
   - Memory footprint, parameter count, file size, and per-flow latency profiling
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_and_algorithms import (
    SelfIoTADNetwork,
    IndependentMarginalPermuter,
    GaussianNoiseAugmenter,
    TabularInfoNCELoss,
    BinaryFocalLoss
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'rt_iot2022_splits.npz')
RESULTS_JSON = os.path.join(BASE_DIR, 'results', 'experiment_results_20reps.json')

SEEDS = [1000 + i * 47 for i in range(20)]

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    macro_f1 = 0.5 * (f1_pos + f1_0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.5
    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except Exception:
        auc_pr = 0.0
    return {
        "precision": round(float(prec * 100), 2),
        "recall": round(float(rec * 100), 2),
        "f1_pos": round(float(f1_pos * 100), 2),
        "macro_f1": round(float(macro_f1 * 100), 2),
        "fpr": round(float(fpr * 100), 2),
        "roc_auc": round(float(auc_roc * 100), 2),
        "pr_auc": round(float(auc_pr * 100), 2),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }

def train_self_iot_ad_pipeline(seed, label_ratio, X_train, y_train, X_val, y_val, X_test, y_test, ablations=None):
    if ablations is None:
        ablations = {}
    no_ssl = ablations.get("no_ssl", False)
    no_contrast = ablations.get("no_contrast", False)
    no_recon = ablations.get("no_recon", False)
    no_teacher = ablations.get("no_teacher", False)
    use_gaussian = ablations.get("use_gaussian", False)
    use_bce = ablations.get("use_bce", False)

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    idx_0 = np.where(y_train == 0)[0]
    idx_1 = np.where(y_train == 1)[0]
    n_0 = max(1, int(len(idx_0) * label_ratio))
    n_1 = max(1, int(len(idx_1) * label_ratio))
    labeled_idx = np.concatenate([
        rng.choice(idx_0, size=n_0, replace=False),
        rng.choice(idx_1, size=n_1, replace=False)
    ])
    rng.shuffle(labeled_idx)

    X_l = torch.tensor(X_train[labeled_idx], dtype=torch.float32)
    y_l = torch.tensor(y_train[labeled_idx], dtype=torch.float32)
    X_u = torch.tensor(X_train, dtype=torch.float32)

    n_val_matched = min(len(X_val), 1771)
    v_idx = rng.choice(len(X_val), size=n_val_matched, replace=False)
    X_v = torch.tensor(X_val[v_idx], dtype=torch.float32)
    y_v = y_val[v_idx]

    model = SelfIoTADNetwork(input_dim=X_train.shape[1])
    augmenter = GaussianNoiseAugmenter(0.20) if use_gaussian else IndependentMarginalPermuter(0.30)

    # Stage 1: SSL Pretraining
    if not no_ssl:
        opt_pre = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit_c = TabularInfoNCELoss(temperature=0.15)
        crit_r = nn.MSELoss()
        u_loader = DataLoader(TensorDataset(X_u), batch_size=256, shuffle=True, drop_last=True)
        model.train()
        for epoch in range(10):
            for (bx,) in u_loader:
                bc, _ = augmenter(bx)
                z_orig = model.encoder(bx)
                proj_orig = F.normalize(model.projector(z_orig), dim=-1)
                proj_corr, recon = model.forward_pretext(bc)
                loss = torch.tensor(0.0)
                if not no_contrast:
                    loss = loss + crit_c(proj_orig, proj_corr)
                if not no_recon:
                    loss = loss + crit_r(recon, bx)
                opt_pre.zero_grad()
                loss.backward()
                opt_pre.step()

    # Pretext frozen checkpoint for linear probing
    pretext_model = copy.deepcopy(model)
    pretext_model.eval()

    # Stage 2: Fine-Tuning
    student = model
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False

    opt_fine = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    if use_bce:
        crit_sup = nn.BCELoss()
    else:
        alpha_w = float((y_train == 0).sum() / len(y_train))
        crit_sup = BinaryFocalLoss(gamma=2.0, alpha=alpha_w)

    l_loader = DataLoader(TensorDataset(X_l, y_l), batch_size=min(32, len(X_l)), shuffle=True)
    u_loader = DataLoader(TensorDataset(X_u), batch_size=256, shuffle=True, drop_last=True)

    best_score = -1.0
    best_state = None
    ema_alpha = 0.996
    tau_conf = 0.85

    for epoch in range(15):
        student.train()
        l_iter = iter(l_loader)
        for (bx_u,) in u_loader:
            try:
                bx_l, by_l = next(l_iter)
            except StopIteration:
                l_iter = iter(l_loader)
                bx_l, by_l = next(l_iter)

            p_l = student.forward_classify(bx_l).squeeze(-1)
            loss_sup = crit_sup(p_l, by_l)

            if not no_teacher:
                with torch.no_grad():
                    target_u = teacher.forward_classify(bx_u).squeeze(-1)
                bx_u_aug, _ = augmenter(bx_u)
                p_u_aug = student.forward_classify(bx_u_aug).squeeze(-1)
                conf_mask = (torch.max(target_u, 1.0 - target_u) > tau_conf).float()
                loss_cons = (conf_mask * (p_u_aug - target_u) ** 2).sum() / (conf_mask.sum() + 1e-6)
                tot_loss = loss_sup + 1.0 * loss_cons
            else:
                tot_loss = loss_sup

            opt_fine.zero_grad()
            tot_loss.backward()
            opt_fine.step()

            if not no_teacher:
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), student.parameters()):
                        tp.data.mul_(ema_alpha).add_(sp.data, alpha=1.0 - ema_alpha)

        student.eval()
        with torch.no_grad():
            val_p = student.forward_classify(X_v).squeeze(-1).numpy()
        vm = compute_all_metrics(y_v, val_p)
        if vm["macro_f1"] > best_score:
            best_score = vm["macro_f1"]
            best_state = copy.deepcopy(student.state_dict())

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()

    X_test_tens = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        test_probs = student.forward_classify(X_test_tens).squeeze(-1).numpy()

    metrics = compute_all_metrics(y_test, test_probs)
    return metrics, test_probs, student, pretext_model, labeled_idx

def run_extended_evaluations():
    print("=" * 70)
    print("Starting Extended Benchmark Validations (20 Repetitions)")
    print("=" * 70)

    data = np.load(DATA_PATH)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    with open(RESULTS_JSON, 'r', encoding='utf-8') as f:
        master = json.load(f)

    # -------------------------------------------------------------------------
    # Part 1: 20-Seed Linear Probe on Frozen Encoder Representations
    # -------------------------------------------------------------------------
    print("\n[Part 1] Running 20-Seed Linear Probe on Representations...")
    lp_runs = []
    lp_pretext_runs = []

    t0 = time.time()
    for idx, s in enumerate(SEEDS):
        rep_key = f"rep_{idx+1}"
        m_fine, p_fine, fine_model, pre_model, l1_idx = train_self_iot_ad_pipeline(
            s, 0.01, X_train, y_train, X_val, y_val, X_test, y_test
        )

        # 1. Linear Probe on fine-tuned representation Z
        fine_model.eval()
        with torch.no_grad():
            Z_tr_fine = fine_model.forward_encoder(torch.tensor(X_train[l1_idx], dtype=torch.float32)).numpy()
            Z_te_fine = fine_model.forward_encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()
        clf_fine = LogisticRegression(max_iter=500, random_state=s).fit(Z_tr_fine, y_train[l1_idx])
        lp_probs = clf_fine.predict_proba(Z_te_fine)[:, 1]
        m_lp = compute_all_metrics(y_test, lp_probs)
        lp_runs.append(m_lp)

        # Update rep_runs entry in master JSON
        if rep_key in master["rep_runs"]:
            master["rep_runs"][rep_key]["Linear_Probe_1pct"] = m_lp

        # 2. Linear Probe on purely unsupervised pretext representation Z_pre
        pre_model.eval()
        with torch.no_grad():
            Z_tr_pre = pre_model.forward_encoder(torch.tensor(X_train[l1_idx], dtype=torch.float32)).numpy()
            Z_te_pre = pre_model.forward_encoder(torch.tensor(X_test, dtype=torch.float32)).numpy()
        clf_pre = LogisticRegression(max_iter=500, random_state=s).fit(Z_tr_pre, y_train[l1_idx])
        lp_pre_probs = clf_pre.predict_proba(Z_te_pre)[:, 1]
        m_lp_pre = compute_all_metrics(y_test, lp_pre_probs)
        lp_pretext_runs.append(m_lp_pre)

        print(f"  Rep {idx+1:02d}/20 [Seed {s}]: Linear Probe Z={m_lp['macro_f1']:.2f}% (FPR={m_lp['fpr']:.2f}%) | Pretext Z_pre={m_lp_pre['macro_f1']:.2f}%")

    lp_f1s = [r["macro_f1"] for r in lp_runs]
    lp_recs = [r["recall"] for r in lp_runs]
    lp_fprs = [r["fpr"] for r in lp_runs]
    lp_aucs = [r["roc_auc"] for r in lp_runs]
    print(f"\nLinear Probe 20-Rep Summary: Macro-F1 = {np.mean(lp_f1s):.2f} +/- {np.std(lp_f1s):.2f}% | Recall = {np.mean(lp_recs):.2f} +/- {np.std(lp_recs):.2f}% | FPR = {np.mean(lp_fprs):.2f} +/- {np.std(lp_fprs):.2f}%")

    master["linear_probe_20reps"] = {
        "mean_macro_f1": round(float(np.mean(lp_f1s)), 2),
        "std_macro_f1": round(float(np.std(lp_f1s)), 2),
        "mean_recall": round(float(np.mean(lp_recs)), 2),
        "std_recall": round(float(np.std(lp_recs)), 2),
        "mean_fpr": round(float(np.mean(lp_fprs)), 2),
        "std_fpr": round(float(np.std(lp_fprs)), 2),
        "mean_roc_auc": round(float(np.mean(lp_aucs)), 2),
        "std_roc_auc": round(float(np.std(lp_aucs)), 2),
        "all_runs": lp_runs,
        "pretext_only_runs": lp_pretext_runs
    }

    # -------------------------------------------------------------------------
    # Part 2: 20-Seed Factorial Ablation Study
    # -------------------------------------------------------------------------
    print("\n[Part 2] Running 20-Seed Factorial Ablation Suite...")
    ab_cfgs = {
        "Full_Self_IoT_AD": {},
        "wo_SSL_Pretraining": {"no_ssl": True},
        "wo_InfoNCE_Contrastive": {"no_contrast": True},
        "wo_Reconstruction_Decoder": {"no_recon": True},
        "wo_Teacher_Consistency": {"no_teacher": True},
        "Gaussian_Noise_Instead_Of_Permutation": {"use_gaussian": True},
        "BCE_Loss_Instead_Of_Focal": {"use_bce": True}
    }
    ab_20_out = {}
    for ab_name, ab_cfg in ab_cfgs.items():
        scores = []
        for s in SEEDS:
            m, _, _, _, _ = train_self_iot_ad_pipeline(
                s, 0.01, X_train, y_train, X_val, y_val, X_test, y_test, ablations=ab_cfg
            )
            scores.append(m["macro_f1"])
        ab_20_out[ab_name] = {
            "mean_f1": round(float(np.mean(scores)), 2),
            "std_f1": round(float(np.std(scores)), 2),
            "scores": [round(x, 2) for x in scores]
        }
        print(f"  20-Seed Ablation [{ab_name}]: {np.mean(scores):.2f} +/- {np.std(scores):.2f}%")

    master["ablations_20reps"] = ab_20_out

    # -------------------------------------------------------------------------
    # Part 3: Edge Quantization & Model Footprint Profiling
    # -------------------------------------------------------------------------
    print("\n[Part 3] Profiling Model Footprint, INT8 Quantization, and Latency...")
    eval_model = SelfIoTADNetwork(input_dim=X_train.shape[1])
    eval_model.eval()

    total_params = sum(p.numel() for p in eval_model.parameters())
    enc_params = sum(p.numel() for p in eval_model.encoder.parameters())
    proj_params = sum(p.numel() for p in eval_model.projector.parameters())
    dec_params = sum(p.numel() for p in eval_model.decoder.parameters())
    clf_params = sum(p.numel() for p in eval_model.classifier.parameters())
    infer_params = enc_params + clf_params

    # Save FP32 model checkpoint
    model_dir = os.path.join(BASE_DIR, 'results', 'models')
    os.makedirs(model_dir, exist_ok=True)
    fp32_path = os.path.join(model_dir, 'self_iot_ad_fp32.pt')
    torch.save(eval_model.state_dict(), fp32_path)
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0

    # Quantize dynamically to INT8 (Linear layers)
    quantized_model = torch.ao.quantization.quantize_dynamic(
        eval_model, {nn.Linear}, dtype=torch.qint8
    )
    int8_path = os.path.join(model_dir, 'self_iot_ad_int8.pt')
    torch.save(quantized_model.state_dict(), int8_path)
    int8_size_kb = os.path.getsize(int8_path) / 1024.0

    # Benchmark single-flow latency (1000 warmup, 5000 timing iterations)
    sample_flow = torch.tensor(X_test[:1], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(500):
            _ = eval_model.forward_classify(sample_flow)
        t_start = time.perf_counter()
        for _ in range(2000):
            _ = eval_model.forward_classify(sample_flow)
        fp32_single_lat_us = ((time.perf_counter() - t_start) / 2000.0) * 1e6

    with torch.no_grad():
        for _ in range(500):
            _ = quantized_model.forward_classify(sample_flow)
        t_start = time.perf_counter()
        for _ in range(2000):
            _ = quantized_model.forward_classify(sample_flow)
        int8_single_lat_us = ((time.perf_counter() - t_start) / 2000.0) * 1e6

    # Batch throughput (batch size 64)
    batch_flows = torch.tensor(X_test[:64], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(100):
            _ = eval_model.forward_classify(batch_flows)
        t_start = time.perf_counter()
        for _ in range(500):
            _ = eval_model.forward_classify(batch_flows)
        batch_fps = (500 * 64) / (time.perf_counter() - t_start)

    edge_profile = {
        "total_parameters": total_params,
        "inference_parameters": infer_params,
        "encoder_parameters": enc_params,
        "classifier_parameters": clf_params,
        "projector_parameters": proj_params,
        "decoder_parameters": dec_params,
        "fp32_checkpoint_kb": round(fp32_size_kb, 2),
        "int8_checkpoint_kb": round(int8_size_kb, 2),
        "fp32_latency_single_flow_us": round(fp32_single_lat_us, 2),
        "int8_latency_single_flow_us": round(int8_single_lat_us, 2),
        "throughput_flows_per_sec": round(batch_fps, 1),
        "compression_ratio": round(fp32_size_kb / (int8_size_kb + 1e-6), 2)
    }
    master["edge_quantization_profile"] = edge_profile
    print("\nEdge Hardware Profiling Results:")
    print(json.dumps(edge_profile, indent=2))

    # Save updated master results
    with open(RESULTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)
    print(f"\n[DONE] Extended benchmarks written to: {RESULTS_JSON}")

if __name__ == '__main__':
    run_extended_evaluations()
