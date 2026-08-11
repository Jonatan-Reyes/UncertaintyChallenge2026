"""Finish a cluster-mode run from saved checkpoints (no retraining).

Reuses the exact stage-4 logic of ``run_cluster`` in train_dinov3_ensemble.py:
combine the saved per-cluster experts on val (plain / temp-scaled average and
cluster-distance soft gate), pick the lowest-val-NLL combination, write
``metrics_val.json`` and ``submission.csv``.

Usage:
    python resume_cluster.py --run-dir /path/to/runs/ensemble_dinov3_raw_clusters_448
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader

import train_dinov3_ensemble as M
from train_ensemble import GATE_TAUS, apply_pipeline, gated_probs
from student.data import IWildCamChallengeDataset
from student.metrics import compute_all_metrics
from student.predict import write_submission

DEFAULT_SPLITS = ["test_public", "test_private"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.run_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    val_ds = IWildCamChallengeDataset(cfg["data_root"], "val", M.eval_transform(cfg["img_size"]))
    val_uids = val_ds.uids
    val_labels = np.asarray(val_ds.labels)
    print(f"val: {len(val_uids)} images, {val_ds.num_classes} classes")

    pipeline = pickle.loads((out_dir / "cluster" / "pipeline.pkl").read_bytes())
    n_clusters = int(cfg["n_clusters"])

    models = []
    for k in range(n_clusters):
        ckpt = out_dir / "experts" / f"expert_seed{k}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        sd = torch.load(ckpt, map_location=device)
        model = M.DinoV3LoraExpert(
            val_ds.num_classes, backbone_id=cfg["backbone"],
            lora_r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            lora_last_layers=int(cfg["lora_last_layers"]),
        ).to(device)
        model.load_state_dict(sd["state_dict"])
        model.eval()
        models.append(model)
    print(f"loaded {len(models)} experts")

    z_val = apply_pipeline(pipeline, Path(cfg["corner_root"]), "val", val_uids, device)
    centers = pipeline["kmeans"].cluster_centers_
    D = cdist(z_val, centers)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
    logits = [M.collect_logits(m, val_loader, device) for m in models]
    stack = torch.stack(logits, dim=0)
    probs_stack = torch.softmax(stack, dim=2).cpu().numpy()

    report = {"plain_average": compute_all_metrics(probs_stack.mean(axis=0), val_labels)}
    chosen = {"name": "plain_average", "tau": None, "probs": probs_stack.mean(axis=0)}

    avg_logits = stack.mean(dim=0)
    T = M.fit_temperature(avg_logits, torch.tensor(val_labels, device=device))
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
    for split in cfg.get("splits", DEFAULT_SPLITS):
        test_ds = IWildCamChallengeDataset(cfg["data_root"], split, M.eval_transform(cfg["img_size"]))
        uids = test_ds.uids
        loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0)
        tlogits = [M.collect_logits(m, loader, device) for m in models]
        tstack = torch.stack(tlogits, dim=0)
        if chosen["name"] == "gated":
            z_test = apply_pipeline(pipeline, Path(cfg["corner_root"]), split, uids, device)
            w = torch.softmax(
                torch.tensor(-cdist(z_test, centers), dtype=torch.float64) / chosen["tau"],
                dim=1,
            ).numpy()
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


if __name__ == "__main__":
    main()
