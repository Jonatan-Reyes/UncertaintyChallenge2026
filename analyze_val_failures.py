"""Find validation images where the model is wrong, ranked by confidence.

Reuses the cached posthoc artifacts of ``ensemble_dinov3_raw_full_448``
(val per-expert logits + frozen-backbone embeddings + fitted UMAP/KMeans
pipeline + chosen cluster-conditional (T, lam, prior) params) and applies the
exact ``cluster_both_k10`` recipe to reconstruct the final val probabilities,
then reports the wrong predictions with the highest max-softmax confidence.

Outputs:
    <run_dir>/posthoc/val_failures.csv        one row per image, all stats
    <run_dir>/posthoc/val_worst_wrong.csv     wrong preds only, sorted by conf
    <run_dir>/posthoc/val_worst_wrong.png     contact sheet of the top-N wrong
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DEFAULT_RUN = Path("/home/alice/work/dtu_ss_26/runs/ensemble_dinov3_raw_full_448")


def softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    x = x / T
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def assign_clusters(pipeline: dict, emb: np.ndarray) -> np.ndarray:
    z = pipeline["umap"].transform(emb)
    return pipeline["kmeans"].predict(z)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--top-n", type=int, default=30,
                        help="number of worst images on the contact sheet")
    args = parser.parse_args()

    out_dir = Path(args.run_dir)
    ph = out_dir / "posthoc"
    cfg = json.loads((out_dir / "config.json").read_text())
    chosen = json.loads((ph / "chosen.json").read_text())

    val_df = pd.read_csv(Path(cfg["data_root"]) / "val" / "labels.csv")
    uids = val_df["uid"].astype(str).tolist()
    labels = val_df["y"].astype(int).to_numpy()
    domains = val_df["domain"].astype(str).to_numpy()
    num_classes = int(json.loads(
        (Path(cfg["data_root"]) / "class_mapping.json").read_text())["num_classes"])

    stack = np.load(ph / "val_logits.npz")["logits"]          # (n_experts, N, C)
    val_emb = np.load(ph / "val_emb.npz")["emb"]              # (N, D)
    avg_logits = stack.mean(axis=0)
    print(f"val: {len(uids)} images | logits {stack.shape} | emb {val_emb.shape}")

    with (ph / "pipeline.pkl").open("rb") as f:
        pipe = pickle.load(f)
    cl = assign_clusters(pipe, val_emb)

    T_s = chosen["params"]["T"]
    la_s = chosen["params"]["lam"]
    pr_s = chosen["params"]["prior"]
    probs = np.empty_like(softmax(avg_logits))
    for k in range(chosen["K"]):
        m = cl == k
        if not m.any():
            continue
        p_k = softmax(avg_logits[m], float(T_s[str(k)]))
        lam = float(la_s[str(k)])
        probs[m] = (1.0 - lam) * p_k + lam * np.asarray(pr_s[str(k)], dtype=np.float64)

    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)
    correct = (preds == labels)
    p_true = probs[np.arange(len(labels)), labels]

    df = pd.DataFrame({
        "uid": uids,
        "domain": domains,
        "y": labels,
        "pred": preds,
        "confidence": confs,
        "p_true": p_true,
        "correct": correct,
        "cluster": cl,
    })
    df["nll"] = -np.log(np.clip(p_true, 1e-12, 1.0))
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    df["brier"] = ((probs - onehot) ** 2).sum(axis=1)

    df.to_csv(ph / "val_failures.csv", index=False)

    worst = df[~df["correct"]].sort_values("confidence", ascending=False)
    worst.to_csv(ph / "val_worst_wrong.csv", index=False)

    n_wrong = int((~df["correct"]).sum())
    print(f"wrong: {n_wrong}/{len(df)} ({n_wrong / len(df):.1%}) | "
          f"id/ood wrong: {(~df['correct'][df['domain']=='id']).sum()}/{df['domain'].eq('id').sum()} vs "
          f"{(~df['correct'][df['domain']=='ood']).sum()}/{df['domain'].eq('ood').sum()}")
    print("worst wrong predictions (top 10):")
    print(worst.head(10).to_string(index=False))

    make_contact_sheet(worst, Path(cfg["data_root"]), args.top_n, ph)


def make_contact_sheet(worst: pd.DataFrame, data_root: Path, top_n: int, ph: Path) -> None:
    top = worst.head(top_n)
    img_dir = data_root / "val" / "images"
    cols = 6
    thumb = 160
    rows = int(np.ceil(len(top) / cols))
    pad = 40
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for i, (_, row) in enumerate(top.iterrows()):
        img = Image.open(img_dir / f"{row['uid']}.jpg").convert("RGB")
        img = img.resize((thumb, thumb))
        r, c = divmod(i, cols)
        x, y = c * thumb, r * (thumb + pad)
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + thumb + 2),
                  f"{row['uid'][:12]}..\n{row['domain']} conf={row['confidence']:.2f}\n"
                  f"y={row['y']} pred={row['pred']}", fill="black", font=font)

    png = ph / "val_worst_wrong.png"
    sheet.save(png)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
