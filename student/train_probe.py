"""Linear probe on cached backbone features (rung 2 of the plan).

Trains only ``nn.Linear(embed_dim, num_classes)`` against features cached by
``student.extract_features`` — no image loading, no backbone forward pass, so
a full run is seconds instead of minutes. Fits a temperature the same way
``student.train`` does, then assembles a full ``Classifier`` (frozen
pretrained backbone + trained head) and saves it through the same
checkpoint format so ``student.eval`` / ``student.predict`` work unmodified.

Usage:
    python -m student.train_probe --features-dir features/dinov2_vitb14_reg4_224 \
        --backbone vit_base_patch14_reg4_dinov2 --out-dir runs/dinov2_vitb14_reg4_224_linear_probe
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from student import metrics as M
from student.model import Classifier
from student.train import fit_temperature, save_checkpoint


def load_split(features_dir: Path, split: str, pooling: str = "embeddings"):
    """``pooling``: 'cls', 'avg', or the legacy single-array key 'embeddings'."""
    d = np.load(features_dir / f"{split}.npz", allow_pickle=True)
    key = "embeddings" if pooling == "embeddings" else f"{pooling}_embeddings"
    embeddings = torch.from_numpy(d[key]).float()
    labels = torch.from_numpy(d["labels"]).long() if "labels" in d else None
    domains = d["domains"] if "domains" in d else None
    uids = d["uids"]
    return embeddings, labels, domains, uids


def evaluate_probe(head: nn.Linear, embeddings: torch.Tensor, labels: torch.Tensor,
                    device, temperature: float = 1.0) -> dict:
    head.eval()
    with torch.no_grad():
        logits = head(embeddings.to(device))
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()
    return M.compute_all_metrics(probs, labels.numpy())


def train_probe(
    features_dir: Path,
    out_dir: Path,
    backbone: str,
    img_size: int,
    pooling: str = "embeddings",
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    extra_train_features_dirs: tuple[Path, ...] = (),
    label_smoothing: float = 0.0,
    brier_weight: float = 0.0,
    seed: int | None = None,
) -> None:
    """``extra_train_features_dirs``: additional feature caches (e.g. an
    aspect-preserving-crop extraction alongside the default squash one) whose
    *train* split gets concatenated onto the training set, so the head learns
    to be invariant to which crop TTA later averages over. Only ``train`` is
    pooled this way -- ``val`` stays single-view since it's used for early
    stopping / temperature fitting against the eval-time distribution.

    ``seed`` controls head init + minibatch order only (data/backbone are
    fixed) -- for training multiple seeds of the same backbone as deep-
    ensemble members."""
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_emb, train_labels, _, _ = load_split(features_dir, "train", pooling)
    for extra_dir in extra_train_features_dirs:
        extra_emb, extra_labels, _, _ = load_split(extra_dir, "train", pooling)
        train_emb = torch.cat([train_emb, extra_emb], dim=0)
        train_labels = torch.cat([train_labels, extra_labels], dim=0)
    val_emb, val_labels, val_domains, _ = load_split(features_dir, "val", pooling)
    num_classes = int(train_labels.max().item()) + 1
    embed_dim = train_emb.shape[1]

    train_emb, train_labels = train_emb.to(device), train_labels.to(device)
    val_emb_dev = val_emb.to(device)

    head = nn.Linear(embed_dim, num_classes).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def criterion(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = ce_criterion(logits, targets)
        if brier_weight > 0:
            probs = torch.softmax(logits, dim=1)
            onehot = torch.nn.functional.one_hot(targets, num_classes).float()
            loss = loss + brier_weight * ((probs - onehot) ** 2).sum(dim=1).mean()
        return loss

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    n = train_emb.shape[0]
    batch_size = 512
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = head(train_emb[idx])
            loss = criterion(logits, train_labels[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_emb_dev)
            val_acc = float((val_logits.argmax(dim=1).cpu() == val_labels).float().mean())

        if epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d} | train_loss={total_loss / n:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"early stopping at epoch {epoch} (no val_acc improvement for {patience} epochs)")
                break

    head.load_state_dict(best_state)

    model_pooling = None if pooling == "embeddings" else pooling
    model = Classifier(
        num_classes, backbone_name=backbone, pretrained=True,
        pooling=model_pooling, img_size=img_size,
    ).to(device)
    model.head.load_state_dict(head.state_dict())
    model.eval()

    hparams = {
        "features_dir": str(features_dir),
        "extra_train_features_dirs": [str(p) for p in extra_train_features_dirs],
        "backbone": backbone,
        "img_size": img_size,
        "pooling": pooling,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "label_smoothing": label_smoothing,
        "brier_weight": brier_weight,
        "seed": seed,
        "method": "linear_probe",
    }
    (out_dir / "config.json").write_text(json.dumps(hparams, indent=2))

    save_checkpoint(model, num_classes, 1.0, out_dir / "model.pt", hyperparameters=hparams)
    print("saved model.pt")

    val_logits_all = head(val_emb_dev).detach()
    T = fit_temperature(val_logits_all, val_labels.to(device))
    save_checkpoint(model, num_classes, T, out_dir / "model_temp_scaled.pt", hyperparameters=hparams)
    print(f"learned T={T:.4f} -> saved model_temp_scaled.pt")

    results = {"T=1.0": evaluate_probe(head, val_emb, val_labels, device, 1.0),
               f"T={T:.4f}": evaluate_probe(head, val_emb, val_labels, device, T)}
    if val_domains is not None:
        for domain in ("id", "ood"):
            mask = val_domains == domain
            if mask.any():
                results[f"T={T:.4f} / {domain}"] = evaluate_probe(
                    head, val_emb[mask], val_labels[mask], device, T
                )
    print(json.dumps(results, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a linear probe on cached features.")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, required=True,
                        help="timm model id used to produce the cached features (needed to rebuild a full Classifier).")
    parser.add_argument("--pooling", type=str, default="embeddings", choices=["cls", "avg", "embeddings"],
                        help="Which cached array to probe. 'embeddings' is the legacy single-array cache format.")
    parser.add_argument("--img-size", type=int, required=True,
                        help="Resolution the features were extracted at (needed to rebuild a matching Classifier/eval transform).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--extra-train-features-dirs", type=Path, nargs="*", default=(),
                        help="Extra feature caches whose train split is concatenated onto training "
                             "(e.g. an aspect-preserving-crop extraction), so the head becomes "
                             "invariant to which view TTA averages over.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--brier-weight", type=float, default=0.0,
                        help="Adds brier_weight * mean-squared-error(probs, onehot) to the CE loss "
                             "-- a composite loss instead of picking one proper scoring rule.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Controls head init + minibatch order, for training multiple seeds "
                             "of the same backbone as deep-ensemble members.")
    args = parser.parse_args()
    train_probe(args.features_dir, args.out_dir, args.backbone, args.img_size, args.pooling,
                args.epochs, args.lr, args.weight_decay, args.patience,
                tuple(args.extra_train_features_dirs), args.label_smoothing, args.brier_weight, args.seed)


if __name__ == "__main__":
    main()
