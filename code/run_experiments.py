"""
End-to-End Experimental Execution Script for the Leakage-Free RT-IoT2022 Benchmark
Runs 20 independent randomized repetitions across unsupervised detectors, tree ensembles,
supervised MLPs, and tabular SSL/semi-supervised architectures (including Mean Teacher and FixMatch),
logging raw test predictions, calibrated metrics, and latency profiles.
"""

import os
import json
import time
import copy
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from model_and_algorithms import (
    IndependentMarginalPermuter, GaussianNoiseAugmenter,
    SelfIoTADNetwork, CapacityMatchedMLP, DeepAutoencoder,
    TabularInfoNCELoss, BinaryFocalLoss, compute_all_metrics
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_20_repetition_benchmark():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    results_dir = os.path.join(base_dir, 'results')
    preds_dir = os.path.join(results_dir, 'raw_predictions')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    splits_file = os.path.join(proc_dir, 'rt_iot2022_splits.npz')
    print(f"Loading data from: {splits_file}")
    data = np.load(splits_file, allow_pickle=True)

    X_train = data['X_train']
    y_train = data['y_train']
    att_train = data['att_train']
    X_val = data['X_val']
    y_val = data['y_val']
    att_val = data['att_val']
    X_test = data['X_test']
    y_test = data['y_test']
    att_test = data['att_test']
    feature_names = data['feature_names'].tolist()

    input_dim = X_train.shape[1]
    n_train = len(X_train)
    n_test = len(X_test)
    print(f"Dataset loaded: Train={n_train:,}, Val={len(X_val):,}, Test={n_test:,}, Features={input_dim}")

    # Generate 20 independent seed runs
    n_reps = 20
    seeds = [1000 + i * 47 for i in range(n_reps)]

    master_results = {
        "metadata": {
            "title": "A Leakage-Free Benchmark of Self-Supervised and Semi-Supervised Representation Learning for IoT Intrusion Detection Under Label Scarcity",
            "benchmark_dataset": "RT-IoT2022 (Deduplicated, 17,710 unique flows)",
            "n_reps": n_reps,
            "seeds": seeds,
            "feature_names": feature_names,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch_version": torch.__version__,
            "device": str(device)
        },
        "rep_runs": {},
        "label_efficiency": {},
        "ablations": {},
        "attack_breakdown": {},
        "latency_profiles": {},
        "linear_probe": {}
    }

    # =========================================================================
    # Latency Benchmarks
    # =========================================================================
    print("\n--- Measuring Single-Flow vs. Amortized Inference Latency ---")
    bench_net = SelfIoTADNetwork(input_dim=input_dim).to(device)
    bench_net.eval()
    dummy_single = torch.randn(1, input_dim, device=device)
    dummy_batch = torch.randn(256, input_dim, device=device)

    for _ in range(50):
        _ = bench_net.forward_classify(dummy_single)
        _ = bench_net.forward_classify(dummy_batch)

    iters = 1000
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = bench_net.forward_classify(dummy_single)
    lat_single_us = ((time.perf_counter() - t0) / iters) * 1e6

    iters_b = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters_b):
            _ = bench_net.forward_classify(dummy_batch)
    lat_batch_us = (((time.perf_counter() - t0) / iters_b) / 256) * 1e6

    print(f"Latency: Single-Flow={lat_single_us:.1f} us | Batched Amortized={lat_batch_us:.2f} us")
    master_results["latency_profiles"] = {
        "single_flow_us": round(lat_single_us, 2),
        "batch_256_amortized_us": round(lat_batch_us, 2)
    }

    # =========================================================================
    # Self-IoT-AD & Deep SSL Training Routine
    # =========================================================================
    def train_self_iot_ad(seed, label_ratio, ablations=None):
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

        # Budget-matched validation subset to prevent leakage
        n_val_matched = min(len(labeled_idx), 100)
        v_idx = rng.choice(len(X_val), size=n_val_matched, replace=False)
        X_v = torch.tensor(X_val[v_idx], dtype=torch.float32, device=device)
        y_v = y_val[v_idx]

        model = SelfIoTADNetwork(input_dim=input_dim).to(device)

        if use_gaussian:
            augmenter = GaussianNoiseAugmenter(sigma=0.20)
        else:
            augmenter = IndependentMarginalPermuter(corruption_ratio=0.30)

        # Pretext
        if not no_ssl:
            opt_pre = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion_contrast = TabularInfoNCELoss(temperature=0.15)
            criterion_recon = nn.MSELoss()
            u_loader = DataLoader(TensorDataset(X_u), batch_size=256, shuffle=True, drop_last=True)

            model.train()
            for epoch in range(10):
                for (batch_x,) in u_loader:
                    batch_x = batch_x.to(device)
                    batch_corr, _ = augmenter(batch_x)

                    z_orig = model.encoder(batch_x)
                    proj_orig = F.normalize(model.projector(z_orig), dim=-1)
                    proj_corr, recon = model.forward_pretext(batch_corr)

                    loss = torch.tensor(0.0, device=device)
                    if not no_contrast:
                        loss = loss + criterion_contrast(proj_orig, proj_corr)
                    if not no_recon:
                        loss = loss + criterion_recon(recon, batch_x)

                    opt_pre.zero_grad()
                    loss.backward()
                    opt_pre.step()

        # Fine-Tuning
        student = model
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad = False

        opt_fine = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
        if use_bce:
            criterion_sup = nn.BCELoss()
        else:
            alpha_weight = float((y_train == 0).sum() / len(y_train))
            criterion_sup = BinaryFocalLoss(gamma=2.0, alpha=alpha_weight)

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

                bx_l, by_l = bx_l.to(device), by_l.to(device)
                bx_u = bx_u.to(device)

                p_l = student.forward_classify(bx_l).squeeze(-1)
                loss_sup = criterion_sup(p_l, by_l)

                if not no_teacher:
                    with torch.no_grad():
                        target_u = teacher.forward_classify(bx_u).squeeze(-1)

                    bx_u_aug, _ = augmenter(bx_u)
                    p_u_aug = student.forward_classify(bx_u_aug).squeeze(-1)

                    conf_mask = (torch.max(target_u, 1.0 - target_u) > tau_conf).float()
                    loss_cons = (conf_mask * (p_u_aug - target_u) ** 2).sum() / (conf_mask.sum() + 1e-6)
                    total_loss = loss_sup + 1.0 * loss_cons
                else:
                    total_loss = loss_sup

                opt_fine.zero_grad()
                total_loss.backward()
                opt_fine.step()

                if not no_teacher:
                    with torch.no_grad():
                        for tp, sp in zip(teacher.parameters(), student.parameters()):
                            tp.data.mul_(ema_alpha).add_(sp.data, alpha=1.0 - ema_alpha)

            student.eval()
            with torch.no_grad():
                val_prob = student.forward_classify(X_v).squeeze(-1).cpu().numpy()
            vm = compute_all_metrics(y_v, val_prob)
            if vm["macro_f1"] > best_score:
                best_score = vm["macro_f1"]
                best_state = copy.deepcopy(student.state_dict())

        if best_state is not None:
            student.load_state_dict(best_state)
        student.eval()

        X_t_tens = torch.tensor(X_test, dtype=torch.float32, device=device)
        with torch.no_grad():
            test_probs = student.forward_classify(X_t_tens).squeeze(-1).cpu().numpy()

        metrics = compute_all_metrics(y_test, test_probs)
        return metrics, test_probs, student

    # =========================================================================
    # FixMatch & Mean Teacher Helpers
    # =========================================================================
    def train_fixmatch(seed, label_ratio):
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        labeled_idx = np.concatenate([
            rng.choice(idx_0, size=max(1, int(len(idx_0) * label_ratio)), replace=False),
            rng.choice(idx_1, size=max(1, int(len(idx_1) * label_ratio)), replace=False)
        ])
        rng.shuffle(labeled_idx)

        X_l = torch.tensor(X_train[labeled_idx], dtype=torch.float32)
        y_l = torch.tensor(y_train[labeled_idx], dtype=torch.float32)
        X_u = torch.tensor(X_train, dtype=torch.float32)

        model = CapacityMatchedMLP(input_dim=input_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit_sup = nn.BCELoss()

        weak_aug = GaussianNoiseAugmenter(sigma=0.05)
        strong_aug = IndependentMarginalPermuter(corruption_ratio=0.30)

        l_loader = DataLoader(TensorDataset(X_l, y_l), batch_size=min(32, len(X_l)), shuffle=True)
        u_loader = DataLoader(TensorDataset(X_u), batch_size=256, shuffle=True, drop_last=True)

        tau_conf = 0.90
        model.train()
        for epoch in range(15):
            l_iter = iter(l_loader)
            for (bx_u,) in u_loader:
                try:
                    bx_l, by_l = next(l_iter)
                except StopIteration:
                    l_iter = iter(l_loader)
                    bx_l, by_l = next(l_iter)

                bx_l, by_l = bx_l.to(device), by_l.to(device)
                bx_u = bx_u.to(device)

                # Weak & strong augmentations
                bx_u_weak, _ = weak_aug(bx_u)
                bx_u_strong, _ = strong_aug(bx_u)

                p_l = model(bx_l).squeeze(-1)
                loss_sup = crit_sup(p_l, by_l)

                with torch.no_grad():
                    p_u_weak = model(bx_u_weak).squeeze(-1)
                    pseudo_label = (p_u_weak >= 0.5).float()
                    conf = torch.max(p_u_weak, 1.0 - p_u_weak)
                    mask = (conf >= tau_conf).float()

                p_u_strong = model(bx_u_strong).squeeze(-1)
                loss_u = (mask * F.binary_cross_entropy(p_u_strong, pseudo_label, reduction='none')).sum() / (mask.sum() + 1e-6)

                total_loss = loss_sup + 1.0 * loss_u
                opt.zero_grad()
                total_loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            test_probs = model(torch.tensor(X_test, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
        return compute_all_metrics(y_test, test_probs)

    # =========================================================================
    # Run 20 Repetitions
    # =========================================================================
    print(f"\n--- Starting {n_reps} Independent Benchmark Repetitions ---")
    t_start = time.time()

    for r_idx, seed in enumerate(seeds):
        rep_key = f"rep_{r_idx+1}"
        rng = np.random.RandomState(seed)

        # Labeled subsets
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        l1_idx = np.concatenate([
            rng.choice(idx_0, size=max(1, int(len(idx_0) * 0.01)), replace=False),
            rng.choice(idx_1, size=max(1, int(len(idx_1) * 0.01)), replace=False)
        ])
        l5_idx = np.concatenate([
            rng.choice(idx_0, size=max(1, int(len(idx_0) * 0.05)), replace=False),
            rng.choice(idx_1, size=max(1, int(len(idx_1) * 0.05)), replace=False)
        ])

        rep_res = {}

        # 1. Isolation Forest (Nominal only)
        X_tr_benign = X_train[y_train == 0]
        sub_ben = rng.choice(len(X_tr_benign), size=min(4000, len(X_tr_benign)), replace=False)
        iso = IsolationForest(n_estimators=100, contamination=0.33, random_state=seed)
        iso.fit(X_tr_benign[sub_ben])
        iso_scores = -iso.score_samples(X_test)
        iso_prob = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
        rep_res["Isolation_Forest"] = compute_all_metrics(y_test, iso_prob)

        # 2. One-Class SVM (Nominal only)
        sub_oc = rng.choice(len(X_tr_benign), size=min(2000, len(X_tr_benign)), replace=False)
        ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        ocsvm.fit(X_tr_benign[sub_oc])
        oc_scores = -ocsvm.score_samples(X_test)
        oc_prob = (oc_scores - oc_scores.min()) / (oc_scores.max() - oc_scores.min() + 1e-9)
        rep_res["One_Class_SVM"] = compute_all_metrics(y_test, oc_prob)

        # 3. Deep Autoencoder (Calibrated threshold: 95th percentile of normal train reconstruction error)
        torch.manual_seed(seed)
        dae = DeepAutoencoder(input_dim=input_dim).to(device)
        dae_opt = torch.optim.AdamW(dae.parameters(), lr=1e-3, weight_decay=1e-4)
        dae_loader = DataLoader(TensorDataset(torch.tensor(X_tr_benign[sub_ben], dtype=torch.float32)), batch_size=256, shuffle=True)
        dae.train()
        for _ in range(10):
            for (bx,) in dae_loader:
                bx = bx.to(device)
                loss = F.mse_loss(dae(bx), bx)
                dae_opt.zero_grad()
                loss.backward()
                dae_opt.step()
        dae.eval()
        with torch.no_grad():
            tr_rec = dae(torch.tensor(X_tr_benign[sub_ben], dtype=torch.float32, device=device))
            tr_err = torch.mean((tr_rec - torch.tensor(X_tr_benign[sub_ben], dtype=torch.float32, device=device))**2, dim=1).cpu().numpy()
            th_95 = np.percentile(tr_err, 95.0)

            te_rec = dae(torch.tensor(X_test, dtype=torch.float32, device=device))
            te_err = torch.mean((te_rec - torch.tensor(X_test, dtype=torch.float32, device=device))**2, dim=1).cpu().numpy()
            dae_prob = np.clip(te_err / (2.0 * th_95 + 1e-6), 0.0, 1.0)
        rep_res["Deep_Autoencoder"] = compute_all_metrics(y_test, dae_prob)

        # 4. Logistic Regression
        lr_1 = LogisticRegression(max_iter=500, random_state=seed).fit(X_train[l1_idx], y_train[l1_idx])
        rep_res["Logistic_Regression_1pct"] = compute_all_metrics(y_test, lr_1.predict_proba(X_test)[:, 1])

        lr_5 = LogisticRegression(max_iter=500, random_state=seed).fit(X_train[l5_idx], y_train[l5_idx])
        rep_res["Logistic_Regression_5pct"] = compute_all_metrics(y_test, lr_5.predict_proba(X_test)[:, 1])

        # 5. Random Forest
        rf_1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed).fit(X_train[l1_idx], y_train[l1_idx])
        rep_res["Random_Forest_1pct"] = compute_all_metrics(y_test, rf_1.predict_proba(X_test)[:, 1])

        rf_5 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed).fit(X_train[l5_idx], y_train[l5_idx])
        rep_res["Random_Forest_5pct"] = compute_all_metrics(y_test, rf_5.predict_proba(X_test)[:, 1])

        # 6. XGBoost
        xgb_1 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric='logloss').fit(X_train[l1_idx], y_train[l1_idx])
        rep_res["XGBoost_1pct"] = compute_all_metrics(y_test, xgb_1.predict_proba(X_test)[:, 1])

        xgb_5 = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed, eval_metric='logloss').fit(X_train[l5_idx], y_train[l5_idx])
        rep_res["XGBoost_5pct"] = compute_all_metrics(y_test, xgb_5.predict_proba(X_test)[:, 1])

        # 7. Capacity-Matched Supervised MLP (Calibrated with AdamW lr=5e-4 & BCE loss)
        for tag, idx_sub in [("1pct", l1_idx), ("5pct", l5_idx)]:
            torch.manual_seed(seed)
            mlp = CapacityMatchedMLP(input_dim=input_dim).to(device)
            mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
            mlp_crit = nn.BCELoss()
            loader = DataLoader(TensorDataset(torch.tensor(X_train[idx_sub], dtype=torch.float32), torch.tensor(y_train[idx_sub], dtype=torch.float32)), batch_size=min(32, len(idx_sub)), shuffle=True)
            mlp.train()
            for _ in range(20):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = mlp(bx).squeeze(-1)
                    loss = mlp_crit(pred, by)
                    mlp_opt.zero_grad()
                    loss.backward()
                    mlp_opt.step()
            mlp.eval()
            with torch.no_grad():
                probs = mlp(torch.tensor(X_test, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
            rep_res[f"Supervised_MLP_{tag}"] = compute_all_metrics(y_test, probs)

        # 8. Mean Teacher Baseline
        mt_1, _, _ = train_self_iot_ad(seed, 0.01, ablations={"no_ssl": True})
        mt_5, _, _ = train_self_iot_ad(seed, 0.05, ablations={"no_ssl": True})
        rep_res["Mean_Teacher_1pct"] = mt_1
        rep_res["Mean_Teacher_5pct"] = mt_5

        # 9. FixMatch Baseline
        rep_res["FixMatch_1pct"] = train_fixmatch(seed, 0.01)
        rep_res["FixMatch_5pct"] = train_fixmatch(seed, 0.05)

        # 10. VIME Baseline
        vime_1, _, _ = train_self_iot_ad(seed, 0.01, ablations={"no_contrast": True, "no_teacher": True})
        vime_5, _, _ = train_self_iot_ad(seed, 0.05, ablations={"no_contrast": True, "no_teacher": True})
        rep_res["VIME_1pct"] = vime_1
        rep_res["VIME_5pct"] = vime_5

        # 11. SCARF Baseline
        scarf_1, _, _ = train_self_iot_ad(seed, 0.01, ablations={"no_recon": True, "no_teacher": True})
        scarf_5, _, _ = train_self_iot_ad(seed, 0.05, ablations={"no_recon": True, "no_teacher": True})
        rep_res["SCARF_1pct"] = scarf_1
        rep_res["SCARF_5pct"] = scarf_5

        # 12. Self-IoT-AD (Full Proposed Framework)
        self_1, probs_self_1, trained_model = train_self_iot_ad(seed, 0.01)
        self_5, probs_self_5, _ = train_self_iot_ad(seed, 0.05)
        rep_res["Self_IoT_AD_1pct"] = self_1
        rep_res["Self_IoT_AD_5pct"] = self_5

        # Linear Probe (RQ1 verification on Seed 1000)
        if r_idx == 0:
            trained_model.eval()
            with torch.no_grad():
                Z_train = trained_model.forward_encoder(torch.tensor(X_train[l1_idx], dtype=torch.float32, device=device)).cpu().numpy()
                Z_test = trained_model.forward_encoder(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
            lp = LogisticRegression(max_iter=500).fit(Z_train, y_train[l1_idx])
            lp_probs = lp.predict_proba(Z_test)[:, 1]
            master_results["linear_probe"] = compute_all_metrics(y_test, lp_probs)

            # Save test prediction arrays for Seed 1000 for ROC/PR curves
            np.savez_compressed(
                os.path.join(preds_dir, 'seed_1000_raw_predictions.npz'),
                y_true=y_test,
                att_test=att_test,
                probs_self_iot_ad_1pct=probs_self_1,
                probs_self_iot_ad_5pct=probs_self_5,
                probs_xgb_1pct=xgb_1.predict_proba(X_test)[:, 1],
                probs_rf_1pct=rf_1.predict_proba(X_test)[:, 1],
                probs_mlp_1pct=rep_res["Supervised_MLP_1pct"],
                probs_iso=iso_prob,
                probs_ocsvm=oc_prob,
                probs_dae=dae_prob
            )

            # Attack type detection breakdown
            atk_breakdown = []
            for atk in np.unique(att_test):
                mask = (att_test == atk)
                cnt = int(mask.sum())
                is_atk = (y_test[mask][0] == 1)
                r1 = float(np.mean(probs_self_1[mask] >= 0.5) * 100) if is_atk else float(np.mean(probs_self_1[mask] < 0.5) * 100)
                r5 = float(np.mean(probs_self_5[mask] >= 0.5) * 100) if is_atk else float(np.mean(probs_self_5[mask] < 0.5) * 100)
                atk_breakdown.append({
                    "Attack_Type": atk,
                    "Category": "Malicious Intrusion" if is_atk else "Benign IoT Baseline",
                    "Test_Flow_Count": cnt,
                    "Detection_Rate_1pct": round(r1, 2),
                    "Detection_Rate_5pct": round(r5, 2)
                })
            master_results["attack_breakdown"] = atk_breakdown

        master_results["rep_runs"][rep_key] = rep_res
        print(f"  Repetition {r_idx+1:02d}/20 [Seed {seed}]: Self-IoT-AD (1%)={self_1['macro_f1']:.2f}% | RF (1%)={rep_res['Random_Forest_1pct']['macro_f1']:.2f}% | XGB (1%)={rep_res['XGBoost_1pct']['macro_f1']:.2f}%")

    print(f"All 20 repetitions completed in {time.time()-t_start:.1f} seconds.")

    # =========================================================================
    # Label Efficiency Sweep (0.5%, 1%, 2%, 5%, 10%) on Seed 1000
    # =========================================================================
    print("\n--- Running Label Efficiency Sweep ---")
    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    b_names = ["0.5%", "1.0%", "2.0%", "5.0%", "10.0%"]
    le_list = []
    for r, b_name in zip(budgets, b_names):
        m_self, _, _ = train_self_iot_ad(1000, r)
        # Supervised MLP at same budget
        torch.manual_seed(1000)
        rng = np.random.RandomState(1000)
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        l_sub = np.concatenate([
            rng.choice(idx_0, size=max(1, int(len(idx_0) * r)), replace=False),
            rng.choice(idx_1, size=max(1, int(len(idx_1) * r)), replace=False)
        ])
        mlp = CapacityMatchedMLP(input_dim=input_dim).to(device)
        mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=5e-4, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(torch.tensor(X_train[l_sub], dtype=torch.float32), torch.tensor(y_train[l_sub], dtype=torch.float32)), batch_size=min(32, len(l_sub)), shuffle=True)
        mlp.train()
        for _ in range(20):
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                loss = F.binary_cross_entropy(mlp(bx).squeeze(-1), by)
                mlp_opt.zero_grad()
                loss.backward()
                mlp_opt.step()
        mlp.eval()
        with torch.no_grad():
            mlp_probs = mlp(torch.tensor(X_test, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
        m_mlp = compute_all_metrics(y_test, mlp_probs)

        le_list.append({
            "Label_Budget": b_name,
            "Labeled_Flows": len(l_sub),
            "Supervised_MLP_F1": round(m_mlp["macro_f1"], 2),
            "Self_IoT_AD_F1": round(m_self["macro_f1"], 2),
            "Absolute_Gain": round(m_self["macro_f1"] - m_mlp["macro_f1"], 2)
        })
        print(f"  Budget {b_name} ({len(l_sub)} flows): MLP={m_mlp['macro_f1']:.2f}% | Self-IoT-AD={m_self['macro_f1']:.2f}%")
    master_results["label_efficiency"] = le_list

    # =========================================================================
    # Factorial Ablations (at 1% labels over 5 seeds)
    # =========================================================================
    print("\n--- Running Factorial Ablations ---")
    ab_cfgs = {
        "Full_Self_IoT_AD": {},
        "wo_SSL_Pretraining": {"no_ssl": True},
        "wo_InfoNCE_Contrastive": {"no_contrast": True},
        "wo_Reconstruction_Decoder": {"no_recon": True},
        "wo_Teacher_Consistency": {"no_teacher": True},
        "Gaussian_Noise_Instead_Of_Permutation": {"use_gaussian": True},
        "BCE_Loss_Instead_Of_Focal": {"use_bce": True}
    }
    ab_out = {}
    for ab_name, ab_cfg in ab_cfgs.items():
        scores = []
        for s in seeds[:5]:
            m, _, _ = train_self_iot_ad(s, 0.01, ablations=ab_cfg)
            scores.append(m["macro_f1"])
        ab_out[ab_name] = {
            "mean_f1": round(float(np.mean(scores)), 2),
            "std_f1": round(float(np.std(scores)), 2),
            "scores": [round(x, 2) for x in scores]
        }
        print(f"  Ablation [{ab_name}]: {np.mean(scores):.2f} +/- {np.std(scores):.2f}%")
    master_results["ablations"] = ab_out

    # Save summary results JSON
    results_json = os.path.join(results_dir, 'experiment_results_20reps.json')
    with open(results_json, 'w', encoding='utf-8') as f:
        json.dump(master_results, f, indent=2)
    print(f"\nSaved 20-repetition benchmark results to: {results_json}")

if __name__ == '__main__':
    run_20_repetition_benchmark()
