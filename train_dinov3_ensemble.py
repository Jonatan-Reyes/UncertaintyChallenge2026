"""Deep ensemble of K full-data DINOv3+LoRA experts.

Strategy change vs ``train_ensemble_dinov3.py``:

The old script split the 18.9k training images into K per-cluster expert
subsets (~1.9k images each). Every LoRA expert overfit immediately
(train_acc > 0.85 while val NLL climbed to ~4.6), so the strong DINOv3
features never got enough data. This version fixes the four things that
were holding the scores down:

1. Full data per expert  -- every expert trains on all train images.
2. Strong augmentation   -- RandomResizedCrop + RandAugment + flip + jitter
   (camera-trap images need it to generalize).
3. Label smoothing + mixup -- 57 noisy classes; smoothing + mixup are the
   standard fixes for overfitting *and* miscalibration.
4. Early stop on val NLL  -- the metric the challenge scores, not accuracy.

Ensemble diversity comes from per-expert seeds (init / shuffle / aug).
Members are combined by uniform averaging of probabilities with a scalar
temperature fit on the averaged logits; a cluster-distance soft gate
(reusing the 64x64-corner pipeline) is also evaluated and whichever
combination has the lowest val NLL is used for the submission.

Native full-image mode (--native):
    Images keep their native resolution (all are 448 high, 560-796 wide), so no
    information is ever downscaled. Each batch is padded to the batch-max (H, W)
    rounded to a multiple of the 16px patch size via ``dynamic_pad_collate``, and
    a width-bucketed sampler keeps within-batch resolution variance low so little
    is wasted. Intensity augmentation (``INTENSITY_JITTER``) is biased toward
    contrast/brightness since ~95% of images are effectively grayscale.

Usage:
    python train_dinov3_ensemble.py [options]
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, inject_adapter_in_model
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import transforms
from tqdm import tqdm

from student.data import IWildCamChallengeDataset, IMAGENET_MEAN, IMAGENET_STD
from student.metrics import compute_all_metrics
from student.predict import write_submission
from student.train import fit_temperature
from train_ensemble import (
    GATE_TAUS,
    apply_pipeline,
    fit_cluster_candidate,
    gated_probs,
    load_corner_arrays,
    pretrained_embeddings,
)

DEFAULT_DATA_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep")
DEFAULT_CORNER_ROOT = Path("/home/alice/work/dtu_ss_26/challenge_data_prep_corners64")
DEFAULT_OUTPUT_DIR = Path("/home/alice/work/dtu_ss_26/runs/ensemble_dinov3_full")

IMG_SIZE = 256
DINOV3_BACKBONE = "vit_base_patch16_dinov3"
LORA_TARGETS = ["qkv", "proj", "fc1", "fc2"]

# Intensity augmentation for the near-monochrome camera-trap data: contrast and
# brightness are the channels that matter on grayscale images; saturation stays
# modest for the ~5-10% faint-color images.
INTENSITY_JITTER = dict(
    brightness=(0.6, 1.4),
    contrast=(0.5, 1.5),
    saturation=(0.5, 1.5),
    hue=0.0,
)
PATCH = 16  # DINOv3 patch size; native batches are padded to a multiple of this


class DinoV3LoraExpert(nn.Module):
    """Frozen DINOv3 backbone + LoRA adapters + linear head."""

    def __init__(
        self,
        num_classes: int,
        backbone_id: str = DINOV3_BACKBONE,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_last_layers: int = 0,
    ):
        super().__init__()
        backbone = timm_create_backbone(backbone_id)
        self.backbone_id = backbone_id
        self.embed_dim = int(backbone.num_features)
        if lora_last_layers > 0:
            n_blocks = len(backbone.blocks)
            targets = [
                name
                for name, mod in backbone.named_modules()
                if isinstance(mod, nn.Linear)
                and any(
                    name.startswith(f"blocks.{bi}.")
                    for bi in range(n_blocks - lora_last_layers, n_blocks)
                )
            ]
        else:
            targets = list(LORA_TARGETS)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
        )
        self.backbone = inject_adapter_in_model(lora_cfg, backbone)
        self.head = nn.Linear(self.embed_dim, int(num_classes))
        self.num_classes = int(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def timm_create_backbone(backbone_id: str) -> nn.Module:
    import timm

    backbone = timm.create_model(backbone_id, pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return backbone


def train_transform(size: int = IMG_SIZE, full_image: bool = False) -> transforms.Compose:
    ops = [
        transforms.Resize((size, size))
        if full_image
        else transforms.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 4 / 3)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(**INTENSITY_JITTER),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def native_train_transform() -> transforms.Compose:
    """Full-image training at native resolution (no resize, no crop)."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(**INTENSITY_JITTER),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def native_eval_transform() -> transforms.Compose:
    """Native-resolution evaluation transform (no resize)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def dynamic_pad_collate(batch):
    """Pad a batch of native-resolution images to the batch max (H, W), rounded
    up to a multiple of ``PATCH``. Handles ``(image, label)`` and ``(image, uid)``
    (test_public has no labels)."""
    imgs, rest = zip(*batch)
    hp = max(int(i.shape[-2]) for i in imgs)
    wp = max(int(i.shape[-1]) for i in imgs)
    hp = int(np.ceil(hp / PATCH) * PATCH)
    wp = int(np.ceil(wp / PATCH) * PATCH)
    out = torch.zeros(len(imgs), int(imgs[0].shape[-3]), hp, wp, dtype=imgs[0].dtype)
    for n, i in enumerate(imgs):
        out[n, :, : i.shape[-2], : i.shape[-1]] = i
    if isinstance(rest[0], str):
        return out, list(rest)
    return out, torch.as_tensor([int(r) for r in rest])


_WIDTH_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _widths_for(ds) -> np.ndarray:
    """Image widths of the base dataset (cached by data root + split)."""
    base = ds.dataset if isinstance(ds, Subset) else ds
    key = (str(base.root), base.split)
    if key not in _WIDTH_CACHE:
        from PIL import Image

        ws = []
        for uid in base.uids:
            for ext in (".jpg", ".jpeg", ".png"):
                p = base.images_dir / f"{uid}{ext}"
                if p.exists():
                    with Image.open(p) as im:
                        ws.append(int(im.size[0]))
                    break
            else:
                raise FileNotFoundError(f"no image for uid {uid}")
        _WIDTH_CACHE[key] = np.asarray(ws, dtype=np.int64)
    return _WIDTH_CACHE[key]


class BucketedShuffleSampler(Sampler):
    """Batches of ``batch_size`` where indices are sorted by width and shuffled
    within windows of ``batch_size * window_mult``. Within-batch max resolution
    therefore tracks the batch itself, so ``dynamic_pad_collate`` wastes little."""

    def __init__(self, ds, batch_size: int, window_mult: int = 4):
        self.bs = batch_size
        self.window = max(batch_size * window_mult, batch_size)
        base_idx = np.asarray(ds.indices) if isinstance(ds, Subset) else np.arange(len(ds))
        self.w = _widths_for(ds)[base_idx]

    def __iter__(self):
        order = np.argsort(self.w, kind="stable")
        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        for start in range(0, len(order), self.window):
            chunk = order[start : start + self.window].tolist()
            rng.shuffle(chunk)
            yield from chunk

    def __len__(self) -> int:
        return len(self.w)


def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        mixed, ya, yb, lam = mixup_data(imgs, labels, mixup_alpha)
        optimizer.zero_grad()
        logits = model(mixed)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return total_loss / total


def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            probs.append(torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy())
            ys.append(np.asarray(labels))
    return compute_all_metrics(np.concatenate(probs), np.concatenate(ys))


def collect_logits(model, loader, device) -> torch.Tensor:
    model.eval()
    chunks = []
    with torch.no_grad():
        for imgs, _ in loader:
            chunks.append(model(imgs.to(device)))
    return torch.cat(chunks, dim=0)


def fit_expert(
    seed: int,
    train_ds,
    val_ds,
    cfg,
    device,
    out_dir,
    hyper,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_ds = Subset(train_ds, train_idx) if train_idx is not None else train_ds
    vl_ds = Subset(val_ds, val_idx) if val_idx is not None else val_ds
    train_loader = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=not cfg.native,
        num_workers=cfg.num_workers, drop_last=False,
        sampler=BucketedShuffleSampler(tr_ds, cfg.batch_size) if cfg.native else None,
        collate_fn=dynamic_pad_collate if cfg.native else None,
    )
    val_loader = DataLoader(
        vl_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=dynamic_pad_collate if cfg.native else None,
    )

    model = DinoV3LoraExpert(
        train_ds.num_classes, backbone_id=cfg.backbone,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        lora_last_layers=cfg.lora_last_layers,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = max(1, cfg.epochs // 10)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print(f"\n=== expert seed {seed}: {sum(p.numel() for p in params):,} trainable params ===")
    best_nll, best_state, no_improve = float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, cfg.mixup_alpha)
        val = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_nll={val['nll']:.4f} val_acc={val['accuracy']:.4f} "
              f"val_brier={val['brier']:.4f}")
        if val["nll"] < best_nll:
            best_nll = val["nll"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"  early stop at epoch {epoch} (best val_nll={best_nll:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = out_dir / "experts" / f"expert_seed{seed}.pt"
    torch.save(
        {"state_dict": model.state_dict(), "num_classes": train_ds.num_classes,
         "seed": int(seed), "hyperparameters": {**hyper, "seed": int(seed)}},
        ckpt,
    )
    print(f"  saved {ckpt} (best val_nll={best_nll:.4f})")
    return model


def run_cluster(cfg, device, out_dir, hyper, train_ds, val_ds) -> None:
    """Cluster hold-out training sets variant: same recipe as the deep ensemble,
    but each expert trains only on its corner-cluster's subset and is combined
    with a cluster-distance soft gate (or plain/temp-scaled average)."""
    n_clusters = cfg.n_clusters
    val_uids = val_ds.uids
    val_labels = np.asarray(val_ds.labels)
    cluster_dir = out_dir / "cluster"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    print("\n[cluster] stage 1: clustering val corner images...")
    cand_pix = fit_cluster_candidate(
        load_corner_arrays(cfg.corner_root, "val", val_uids), n_clusters, cfg.base_seed
    )
    cand_pix["method"] = "pixels"
    print(f"  pixels:     silhouette={cand_pix['silhouette']:.4f}")

    cand_emb = fit_cluster_candidate(
        pretrained_embeddings(val_uids, cfg.corner_root, "val", cfg.backbone, device),
        n_clusters,
        cfg.base_seed,
    )
    cand_emb["method"] = "pretrained"
    cand_emb["backbone"] = cfg.backbone
    print(f"  pretrained: silhouette={cand_emb['silhouette']:.4f}")

    pipeline = cand_pix if cand_pix["silhouette"] >= cand_emb["silhouette"] else cand_emb
    print(f"  -> using {pipeline['method']} features (silhouette={pipeline['silhouette']:.4f})")
    val_assign = np.asarray(pipeline["labels"])
    print("  val cluster sizes:", np.bincount(val_assign, minlength=n_clusters).tolist())
    with open(cluster_dir / "pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("\n[cluster] stage 2: routing training images to clusters...")
    train_uids = train_ds.uids
    z_train = apply_pipeline(pipeline, cfg.corner_root, "train", train_uids, device)
    train_assign = pipeline["kmeans"].predict(z_train)
    print("  train cluster sizes:", np.bincount(train_assign, minlength=n_clusters).tolist())

    print("\n[cluster] stage 3: training cluster experts...")
    full_val_idx = np.arange(len(val_ds))
    models = []
    for k in range(n_clusters):
        train_idx = np.where(train_assign == k)[0]
        val_idx = np.where(val_assign == k)[0]
        if len(train_idx) == 0:
            print(f"  expert {k}: empty training set, skipping")
            models.append(None)
            continue
        es_val_idx = val_idx if len(val_idx) >= 20 else full_val_idx
        if len(val_idx) < 20:
            print(f"  (val subset {len(val_idx)} < 20 -> early stopping on full val)")
        models.append(fit_expert(k, train_ds, val_ds, cfg, device, out_dir, hyper,
                                 train_idx=train_idx, val_idx=es_val_idx))
    torch.cuda.empty_cache()

    print("\n[cluster] stage 4: combining experts on val...")
    z_val = apply_pipeline(pipeline, cfg.corner_root, "val", val_uids, device)
    centers = pipeline["kmeans"].cluster_centers_
    D = cdist(z_val, centers)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        collate_fn=dynamic_pad_collate if cfg.native else None,
    )
    logits = []
    for m in models:
        if m is None:
            logits.append(torch.zeros(len(val_ds), val_ds.num_classes, device=device))
        else:
            logits.append(collect_logits(m, val_loader, device))
    stack = torch.stack(logits, dim=0)
    probs_stack = torch.softmax(stack, dim=2).cpu().numpy()

    report = {"plain_average": compute_all_metrics(probs_stack.mean(axis=0), val_labels)}
    chosen = {"name": "plain_average", "tau": None, "probs": probs_stack.mean(axis=0)}

    avg_logits = stack.mean(dim=0)
    T = fit_temperature(avg_logits, torch.tensor(val_labels, device=device))
    avg_T = torch.softmax(avg_logits / T, dim=1).cpu().numpy()
    report["ensemble_T"] = float(T)
    report["temp_scaled_average"] = compute_all_metrics(avg_T, val_labels)
    if report["temp_scaled_average"]["nll"] < report["plain_average"]["nll"]:
        chosen = {"name": "temp_scaled_average", "tau": None, "probs": avg_T}

    best_gate = None
    for tau in GATE_TAUS:
        w = torch.softmax(torch.tensor(-D, dtype=torch.float64) / tau, dim=1).numpy()
        g = gated_probs(probs_stack, w)
        m = compute_all_metrics(g, val_labels)
        if best_gate is None or m["nll"] < best_gate[1]["nll"]:
            best_gate = (tau, m, w, g)
    report["gate_tau"] = best_gate[0]
    report["gated"] = best_gate[1]
    if best_gate[1]["nll"] < report[chosen["name"]]["nll"]:
        chosen = {"name": "gated", "tau": best_gate[0], "probs": best_gate[3]}

    if val_ds.domains is not None:
        dom_arr = np.asarray(val_ds.domains)
        report["by_domain"] = {}
        for dom in sorted(set(dom_arr)):
            mask = dom_arr == dom
            report["by_domain"][dom] = compute_all_metrics(chosen["probs"][mask], val_labels[mask])
    report["chosen"] = chosen["name"]
    (out_dir / "metrics_val.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    all_uids, all_probs = [], []
    eval_tf = native_eval_transform() if cfg.native else eval_transform(cfg.img_size)
    for split in cfg.splits:
        test_ds = IWildCamChallengeDataset(cfg.data_root, split, eval_tf)
        uids = test_ds.uids
        loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
            collate_fn=dynamic_pad_collate if cfg.native else None,
        )
        tlogits = []
        for m in models:
            if m is None:
                tlogits.append(torch.zeros(len(uids), test_ds.num_classes, device=device))
            else:
                tlogits.append(collect_logits(m, loader, device))
        tstack = torch.stack(tlogits, dim=0)
        if chosen["name"] == "gated":
            z_test = apply_pipeline(pipeline, cfg.corner_root, split, uids, device)
            w = torch.softmax(torch.tensor(-cdist(z_test, centers), dtype=torch.float64) / chosen["tau"],
                              dim=1).numpy()
            tprobs = gated_probs(torch.softmax(tstack, dim=2).cpu().numpy(), w)
        else:
            tmean = tstack.mean(dim=0)
            if chosen["name"] == "temp_scaled_average":
                tmean = tmean / T
            tprobs = torch.softmax(tmean, dim=1).cpu().numpy()
        all_uids.extend(uids)
        all_probs.append(tprobs)

    submission = out_dir / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), submission)
    print(f"\nwrote {submission} (chosen={chosen['name']}, T={T:.4f}, gate_tau={report['gate_tau']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--corner-root", type=Path, default=DEFAULT_CORNER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--backbone", type=str, default=DINOV3_BACKBONE)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--full-image", action="store_true",
                        help="use the whole image at --img-size (Resize, no random-resized crop)")
    parser.add_argument("--native", action="store_true",
                        help="keep images at native resolution (no resize); pad each batch to the "
                             "batch max with a width-bucketed sampler and dynamic collate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--lora-last-layers", type=int, default=0,
                        help="LoRA only the last N transformer blocks (0 = all blocks)")
    parser.add_argument("--cluster-mode", type=str, choices=["off", "on"], default="off",
                        help="'on' = per-cluster hold-out training sets from corner clustering + gating")
    parser.add_argument("--no-gate", action="store_true",
                        help="skip the cluster-distance soft-gate comparison")
    parser.add_argument("--splits", nargs="+", default=["test_public", "test_private"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    (out_dir / "experts").mkdir(parents=True, exist_ok=True)

    hyper = {
        "n_experts": int(args.n_experts),
        "backbone": args.backbone,
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "img_size": int(args.img_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "mixup_alpha": float(args.mixup_alpha),
        "patience": int(args.patience),
        "base_seed": int(args.base_seed),
        "lora_last_layers": int(args.lora_last_layers),
        "cluster_mode": str(args.cluster_mode),
        "full_image": bool(args.full_image),
        "native": bool(args.native),
        "warmup_epochs": max(1, args.epochs // 10),
        "n_clusters": int(args.n_clusters),
        "data_root": str(args.data_root),
        "corner_root": str(args.corner_root),
    }
    (out_dir / "config.json").write_text(json.dumps(hyper, indent=2))

    train_ds = IWildCamChallengeDataset(
        args.data_root, "train",
        native_train_transform() if args.native else train_transform(args.img_size, args.full_image)
    )
    val_ds = IWildCamChallengeDataset(
        args.data_root, "val",
        native_eval_transform() if args.native else eval_transform(args.img_size)
    )
    val_labels = np.asarray(val_ds.labels)
    print(f"train: {len(train_ds)} images, {train_ds.num_classes} classes | "
          f"val: {len(val_ds)} images")

    if args.cluster_mode == "on":
        run_cluster(args, device, out_dir, hyper, train_ds, val_ds)
        return

    # ---- Stage 1: train K full-data experts ----
    models = []
    for i in range(args.n_experts):
        seed = args.base_seed + i
        models.append(fit_expert(seed, train_ds, val_ds, args, device, out_dir, hyper))
    torch.cuda.empty_cache()

    # ---- Stage 2: ensemble combinations on val ----
    print("\n[stage 2] combining experts on val...")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=dynamic_pad_collate if args.native else None)
    logits = [collect_logits(m, val_loader, device) for m in models]
    avg_logits = torch.stack(logits).mean(dim=0)
    T = fit_temperature(avg_logits, torch.tensor(val_labels, device=device))
    avg_probs = torch.softmax(avg_logits, dim=1).cpu().numpy()
    avg_T_probs = torch.softmax(avg_logits / T, dim=1).cpu().numpy()

    report = {"ensemble_T": float(T)}
    report["plain_average"] = compute_all_metrics(avg_probs, val_labels)
    report["temp_scaled_average"] = compute_all_metrics(avg_T_probs, val_labels)
    chosen = {"name": "temp_scaled_average", "tau": None, "probs": avg_T_probs}
    if report["plain_average"]["nll"] < report["temp_scaled_average"]["nll"]:
        chosen = {"name": "plain_average", "tau": None, "probs": avg_probs}

    gate_pipeline = None
    if not args.no_gate:
        print("\n  evaluating cluster-distance soft gate...")
        val_uids = val_ds.uids
        cand = fit_cluster_candidate(
            load_corner_arrays(args.corner_root, "val", val_uids), args.n_clusters, args.base_seed
        )
        gate_pipeline = {**cand, "method": "pixels"}
        z_val = apply_pipeline(gate_pipeline, args.corner_root, "val", val_uids, device)
        centers = gate_pipeline["kmeans"].cluster_centers_
        D = cdist(z_val, centers)
        probs_stack = np.stack([torch.softmax(l, dim=1).cpu().numpy() for l in logits], axis=0)
        best_gate = None
        for tau in GATE_TAUS:
            w = torch.softmax(torch.tensor(-D, dtype=torch.float64) / tau, dim=1).numpy()
            g = gated_probs(probs_stack, w)
            m = compute_all_metrics(g, val_labels)
            if best_gate is None or m["nll"] < best_gate[1]["nll"]:
                best_gate = (tau, m, w, g)
        report["gate_tau"] = best_gate[0]
        report["gated"] = best_gate[1]
        if best_gate[1]["nll"] < report[chosen["name"]]["nll"]:
            chosen = {"name": "gated", "tau": best_gate[0], "probs": best_gate[3]}

    if val_ds.domains is not None:
        dom_arr = np.asarray(val_ds.domains)
        report["by_domain"] = {}
        for dom in sorted(set(dom_arr)):
            mask = dom_arr == dom
            report["by_domain"][dom] = compute_all_metrics(chosen["probs"][mask], val_labels[mask])

    report["chosen"] = chosen["name"]
    (out_dir / "metrics_val.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # ---- Stage 3: submission on test splits ----
    all_uids, all_probs = [], []
    eval_tf = native_eval_transform() if args.native else eval_transform(args.img_size)
    for split in args.splits:
        test_ds = IWildCamChallengeDataset(args.data_root, split, eval_tf)
        uids = test_ds.uids
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=dynamic_pad_collate if args.native else None)
        tlogits = [collect_logits(m, loader, device) for m in models]
        if chosen["name"] == "gated":
            z_test = apply_pipeline(gate_pipeline, args.corner_root, split, uids, device)
            w = torch.softmax(
                torch.tensor(-cdist(z_test, gate_pipeline["kmeans"].cluster_centers_),
                             dtype=torch.float64) / chosen["tau"], dim=1).numpy()
            probs_stack = np.stack(
                [torch.softmax(l, dim=1).cpu().numpy() for l in tlogits], axis=0
            )
            test_probs = gated_probs(probs_stack, w)
        else:
            test_probs = torch.softmax(torch.stack(tlogits).mean(dim=0) / T, dim=1).cpu().numpy()
        all_uids.extend(uids)
        all_probs.append(test_probs)

    submission = out_dir / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), submission)
    print(f"\nwrote {submission} (chosen={chosen['name']}, T={T:.4f})")


if __name__ == "__main__":
    main()
