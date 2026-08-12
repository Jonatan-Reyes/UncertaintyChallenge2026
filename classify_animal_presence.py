"""Ask a small VLM whether each camera-trap image contains an animal.

Runs a small vision-language model (default: google/gemma-3-4b-it, 4B) over the
train/val images and writes, per image:

    uid, y, domain, has_animal, answer_raw, parse

where ``has_animal = 1`` when the model says "There is an animal" and ``0`` when
it says "There is NOT an animal".  The result CSV can then be used to relabel /
prune images that contain no animal (many camera-trap images are empty).

Designed for long runs:
- checkpointing: results are written every ``--save-every`` batches and ``--resume``
  skips uids already scored, so you can interrupt and restart safely.
- ``--limit N`` runs only the first N images per split (smoke test).

Usage:
    python classify_animal_presence.py \
        --data-root /home/alice/work/dtu_ss_26/challenge_data \
        --output-dir runs/vlm_animal_labels \
        --batch-size 8

    python classify_animal_presence.py --limit 3            # smoke test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from student.data import IWildCamChallengeDataset

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    sys.exit("transformers not installed (conda env `ss`)")

ANIMAL_PHRASE = "there is an animal"
NO_ANIMAL_PHRASE = "there is not an animal"

PROMPT = (
    "You are reviewing a motion-triggered camera trap image from a wildlife "
    "monitoring station. Is there an animal in this image? "
    "Answer with EXACTLY one of the two following phrases and nothing else: "
    "'There is an animal' or 'There is NOT an animal'. "
    "If you are not sure, answer 'There is an animal'."
)


def parse_answer(answer: str) -> int | None:
    a = answer.lower().replace("\n", " ")
    if NO_ANIMAL_PHRASE in a:
        return 0
    if ANIMAL_PHRASE in a:
        return 1
    return None


def run_split(
    split: str,
    ds: IWildCamChallengeDataset,
    processor,
    model,
    device,
    args,
    out_dir: Path,
) -> None:
    out_csv = out_dir / f"animal_presence_{split}.csv"
    rows: list[dict] = []
    done_uids: set[str] = set()
    if args.resume and out_csv.exists():
        done = pd.read_csv(out_csv)
        done_uids = set(done["uid"].astype(str))
        rows = done.to_dict("records")
        print(f"  resume: {len(done_uids)} already scored for {split}")

    uids = ds.uids
    labels = np.asarray(ds.labels)
    domains = np.asarray(ds.domains) if ds.domains else None
    if args.limit is not None:
        uids = uids[:args.limit]
        labels = labels[:args.limit]
        if domains is not None:
            domains = domains[:args.limit]

    img_dir = ds.images_dir
    n_initial = len(rows)
    n_unclear = 0
    t0 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    t1 = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if t0 is not None:
        t0.record()

    for bi in range(0, len(uids), args.batch_size):
        chunk = uids[bi:bi + args.batch_size]
        todo = [i for i, u in enumerate(chunk) if u not in done_uids]
        if not todo:
            continue

        pil_imgs = []
        for i in todo:
            u = chunk[i]
            img = Image.open(img_dir / f"{u}.jpg").convert("RGB")
            img.thumbnail((args.image_size, args.image_size), Image.LANCZOS)
            pil_imgs.append(img)

        msgs = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT},
            ],
        }] for img in pil_imgs]

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            inputs = processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                processor_kwargs={
                    "padding": True,
                },
            ).to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        generated = out[:, inputs["input_ids"].shape[1]:]
        answers = processor.batch_decode(generated, skip_special_tokens=True)

        for j, i in enumerate(todo):
            u = chunk[i]
            ans = answers[j]
            has_animal = parse_answer(ans)
            if has_animal is None:
                n_unclear += 1
                has_animal = 1  # be conservative: keep the image as "animal present"
            gi = bi + i
            rows.append({
                "uid": u,
                "y": int(labels[gi]),
                "domain": str(domains[gi]) if domains is not None else "",
                "has_animal": has_animal,
                "answer_raw": ans.strip(),
                "parse": "ok" if parse_answer(ans) is not None else "unclear",
            })

        if (bi + 1) % args.save_every == 0:
            write_csv(out_csv, rows, done_uids)

    write_csv(out_csv, rows, done_uids)

    if t0 is not None:
        t1.record(); torch.cuda.synchronize()
        n_new = len(rows) - n_initial
        sec_per_img = t0.elapsed_time(t1) / 1000 / n_new if n_new else 0.0
        print(f"  {split}: {n_new} scored this run ({len(rows)} total), {n_unclear} unclear "
              f"({sec_per_img:.2f} s/img -> est {sec_per_img*len(uids)/3600:.1f} h for full split)")

    has = np.array([r["has_animal"] for r in rows])
    print(f"  {split}: animal-present {has.sum()} / no-animal {(~has.astype(bool)).sum()} "
          f"/ total {len(has)}")

    if args.relabeled_out is not None and len(rows):
        relabel_split(out_csv, Path(args.relabeled_out), split, len(uids))


def relabel_split(out_csv: Path, out_root: Path, split: str, n_total: int) -> None:
    """Write <out_root>/<split>/labels_relabeled.csv with only animal-present rows."""
    df = pd.read_csv(out_csv)
    kept = df[df["has_animal"] == 1][["uid", "y"]].copy()
    if "domain" in df.columns:
        kept = df[df["has_animal"] == 1][["uid", "y", "domain"]]
    dest = out_root / split
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest / "labels_relabeled.csv"
    kept.to_csv(dest, index=False)
    dropped = int((df["has_animal"] == 0).sum())
    print(f"  relabel: kept {len(kept)}/{n_total}, dropped {dropped} -> {dest}")


def write_csv(out_csv: Path, rows: list[dict], done_uids: set[str]) -> None:
    cols = ["uid", "y", "domain", "has_animal", "answer_raw", "parse"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    done_uids.update(r["uid"] for r in rows)
    print(f"  saved {out_csv} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/alice/work/dtu_ss_26/challenge_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/alice/work/dtu_ss_26/runs/vlm_animal_labels"))
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=896,
                        help="thumbnail side before the processor (gemma-3 vision tower is fixed at 896; "
                             "lower this for models like SmolVLM that accept smaller inputs)")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--dtype", choices=["bf16", "float32"], default="bf16")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N images per split (smoke test)")
    parser.add_argument("--relabeled-out", type=Path, default=None,
                        help="if set, write <split>/labels_relabeled.csv per split keeping only "
                             "animal-present images (drop no-animal ones)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model, token=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch_dtype, token=True,
    ).to(device)
    model.eval()

    for split in args.splits:
        print(f"\n=== {split} ===")
        ds = IWildCamChallengeDataset(args.data_root, split, transform=None)
        run_split(split, ds, processor, model, device, args, out_dir)

    print("\ndone.")


if __name__ == "__main__":
    main()
