"""
Core Neural Architectures, Pretext Objectives, Semi-Supervised Algorithms,
and Baselines for the Leakage-Free RT-IoT2022 Benchmark.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# =============================================================================
# Augmenters
# =============================================================================

class IndependentMarginalPermuter:
    """
    Independent Marginal Permutation Augmenter (SCARF-style).
    Replaces a random subset of continuous features with values sampled
    independently per feature column from the empirical mini-batch marginals.
    """
    def __init__(self, corruption_ratio: float = 0.30):
        self.corruption_ratio = corruption_ratio

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_features = x.shape
        mask = (torch.rand(batch_size, num_features, device=x.device) > self.corruption_ratio).float()

        x_corrupt = x.clone()
        for j in range(num_features):
            perm_idx = torch.randperm(batch_size, device=x.device)
            surrogate_col = x[perm_idx, j]
            x_corrupt[:, j] = mask[:, j] * x[:, j] + (1.0 - mask[:, j]) * surrogate_col

        return x_corrupt, mask


class GaussianNoiseAugmenter:
    """Alternative baseline augmenter: zero-mean additive Gaussian perturbation."""
    def __init__(self, sigma: float = 0.20):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x) * self.sigma
        return x + noise, torch.ones_like(x)


# =============================================================================
# Neural Architectures
# =============================================================================

class SelfIoTADNetwork(nn.Module):
    """
    Dual-Task Neural Architecture for Self-IoT-AD:
    - Encoder f_theta: 21 -> 128 -> 64 (11,456 parameters)
    - Projector g_psi: 64 -> 64 -> 32 (6,240 parameters)
    - Decoder r_phi: 64 -> 128 -> 21 (11,029 parameters)
    - Classifier c_omega: 64 -> 32 -> 1 (2,113 parameters)
    Total pretraining parameters: 30,838.
    Total inference parameters: 13,569.
    """
    def __init__(self, input_dim: int = 21, hidden_dim: int = 128, embed_dim: int = 64, proj_dim: int = 32):
        super(SelfIoTADNetwork, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Encoder f_theta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.2)
        )

        # Contrastive Projection Head g_psi
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )

        # Continuous Denoising Decoder r_phi
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Classifier Head c_omega
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_pretext(self, x_corrupt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x_corrupt)
        proj = F.normalize(self.projector(z), dim=-1)
        recon = self.decoder(z)
        return proj, recon

    def forward_classify(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        logits = self.classifier(z)
        return torch.sigmoid(logits)


class CapacityMatchedMLP(nn.Module):
    """
    Capacity-matched Supervised MLP baseline:
    Same 21 -> 128 -> 64 -> 32 -> 1 architecture (13,569 parameters),
    trained from scratch with identical capacity to Self-IoT-AD inference.
    """
    def __init__(self, input_dim: int = 21, hidden_dim: int = 128, embed_dim: int = 64):
        super(CapacityMatchedMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return torch.sigmoid(self.classifier(z))


class DeepAutoencoder(nn.Module):
    """Unsupervised Deep Autoencoder: 21 -> 64 -> 16 -> 64 -> 21 (5,029 params)."""
    def __init__(self, input_dim: int = 21, hidden_dim: int = 64, latent_dim: int = 16):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# =============================================================================
# Losses
# =============================================================================

class TabularInfoNCELoss(nn.Module):
    """InfoNCE Mutual Information Loss for Tabular Contrastive Learning."""
    def __init__(self, temperature: float = 0.15):
        super(TabularInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_orig: torch.Tensor, proj_corrupt: torch.Tensor) -> torch.Tensor:
        batch_size = proj_orig.shape[0]
        sim_matrix = torch.matmul(proj_orig, proj_corrupt.T) / self.temperature
        labels = torch.arange(batch_size, device=proj_orig.device)
        return F.cross_entropy(sim_matrix, labels)


class BinaryFocalLoss(nn.Module):
    """Focal Loss with focusing exponent gamma and optional class weighting."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        pt = torch.where(target == 1, p, 1.0 - p)
        focal_weight = (1.0 - pt) ** self.gamma
        bce = -torch.log(pt)

        if self.alpha is not None:
            alpha_t = torch.where(target == 1, self.alpha, 1.0 - self.alpha)
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce
        return loss.mean()


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Computes mathematically verified, consistent evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    # Macro-F1 across both benign (0) and attack (1)
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_neg = (2 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    macro_f1 = 0.5 * (f1_pos + f1_neg)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = 0.5

    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec * 100),
        "recall": float(rec * 100),
        "f1_pos": float(f1_pos * 100),
        "macro_f1": float(macro_f1 * 100),
        "fpr": float(fpr * 100),
        "roc_auc": float(roc_auc * 100),
        "pr_auc": float(pr_auc * 100)
    }
