"""Build a submission from the expert checkpoints that already exist in a run dir.

No retraining. Loads every ``expert_seed*.pt`` currently saved in ``--run-dir``,
averages their logits on val (plain + temp-scaled), picks the lower-val-NLL
combination, and writes ``metrics_val.json`` and ``submission.csv``.

This is the same stage-4 combine logic as ``run_cluster`` (minus the cluster
gate, which needs a saved embedder), intended for an interrupted cluster run
where only some experts finished.

Usage:
    python make_submission_partial.py --run-dir /path/to/runs/..._cluster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_dinov3_ensemble as M
from student.data import IWildCamChallengeDataset
from student.metrics import compute_all_metrics
from student.predict import write_submission

DEFAULT_SPLITS = ["test_public", "test_private"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="run dir with config.json and experts/expert_seed*.pt")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the data_root recorded in config.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.run_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = args.data_root if args.data_root is not None else cfg["data_root"]

    val_ds = IWildCamChallengeDataset(data_root, "val", M.eval_transform(cfg["img_size"]))
    val_uids = val_ds.uids
    val_labels = np.asarray(val_ds.labels)
    print(f"val: {len(val_uids)} images, {val_ds.num_classes} classes")

    expert_dir = out_dir / "experts"
    ckpts = sorted(expert_dir.glob("expert_seed*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no expert_seed*.pt found in {expert_dir}")
    print(f"found {len(ckpts)} experts: {[p.name for p in ckpts]}")

    models = []
    for ckpt in ckpts:
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

    amp = bool(cfg.get("amp_bf16", False))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=0)
    logits = [M.collect_logits(m, val_loader, device, amp) for m in models]
    stack = torch.stack(logits, dim=0)
    avg_logits = stack.mean(dim=0)
    probs = torch.softmax(avg_logits, dim=1).cpu().numpy()

    report = {"n_experts": len(models), "experts": [p.name for p in ckpts]}
    report["plain_average"] = compute_all_metrics(probs, val_labels)
    chosen = {"name": "plain_average", "probs": probs}

    T = M.fit_temperature(avg_logits, torch.tensor(val_labels, device=device))
    avg_T = torch.softmax(avg_logits / T, dim=1).cpu().numpy()
    report["ensemble_T"] = float(T)
    report["temp_scaled_average"] = compute_all_metrics(avg_T, val_labels)
    if report["temp_scaled_average"]["nll"] < report["plain_average"]["nll"]:
        chosen = {"name": "temp_scaled_average", "probs": avg_T}

    if val_ds.domains is not None:
        dom_arr = np.asarray(val_ds.domains)
        report["by_domain"] = {}
        for dom in sorted(set(dom_arr)):
            mask = dom_arr == dom
            report["by_domain"][dom] = compute_all_metrics(chosen["probs"][mask],
                                                           val_labels[mask])
    report["chosen"] = chosen["name"]
    (out_dir / "metrics_val.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    all_uids, all_probs = [], []
    for split in cfg.get("splits", DEFAULT_SPLITS):
        test_ds = IWildCamChallengeDataset(data_root, split, M.eval_transform(cfg["img_size"]))
        uids = test_ds.uids
        loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=0)
        tlogits = [M.collect_logits(m, loader, device, amp) for m in models]
        tmean = torch.stack(tlogits, dim=0).mean(dim=0)
        if chosen["name"] == "temp_scaled_average":
            tmean = tmean / T
        tprobs = torch.softmax(tmean, dim=1).cpu().numpy()
        all_uids.extend(uids)
        all_probs.append(tprobs)
        print(f"{split}: {len(uids)} images")

    submission = out_dir / "submission.csv"
    write_submission(all_uids, np.concatenate(all_probs, axis=0), submission)
    print(f"\nwrote {submission} (chosen={chosen['name']}, T={T:.4f})")


if __name__ == "__main__":
    main()
