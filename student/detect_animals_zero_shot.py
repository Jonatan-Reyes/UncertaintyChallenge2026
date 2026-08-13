"""Detect whether each challenge image contains any animal using Grounding DINO.

This writes a compact auxiliary CSV with one row per image:

    uid, animal_present, max_confidence

The default prompt is deliberately generic (``animal.``), so the model is used
only as an animal/no-animal detector rather than as a species classifier.

Example:
    python -m student.detect_animals_zero_shot \
        --data-root challenge_data \
        --out-dir zero_shot_animals \
        --model-size tiny \
        --splits train val test_public test_private
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test_public", "test_private")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_SIZES = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


@dataclass(frozen=True)
class ImageRecord:
    uid: str
    path: Path
    split: str


def discover_images(data_root: Path, splits: Iterable[str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in splits:
        images_dir = data_root / split / "images"
        if not images_dir.exists():
            print(f"skipping {split}: {images_dir} does not exist")
            continue
        split_records = [
            ImageRecord(uid=path.stem, path=path, split=split)
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        records.extend(split_records)
    if not records:
        raise FileNotFoundError(f"found no images under {data_root} for splits {tuple(splits)}")
    return records


class GroundingDinoAnimalDetector:
    def __init__(
        self,
        model_id: str,
        device: Any,
        box_threshold: float,
        text_threshold: float,
        prompt: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Grounding DINO detection requires torch and transformers. Install them with "
                "`pip install -r student/requirements.txt`."
            ) from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(device).eval()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt = prompt if prompt.endswith(".") else f"{prompt}."

    def detect_max_confidence(self, image_path: Path) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with self.torch.no_grad():
            with self.torch.autocast(
                device_type="cuda",
                dtype=self.torch.float16,
                enabled=self.device.type == "cuda",
            ):
                outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]
        result = self._post_process(outputs, inputs, target_sizes)[0]
        scores = result.get("scores")
        if scores is None or len(scores) == 0:
            return 0.0
        return float(scores.max().detach().cpu().item())

    def _post_process(self, outputs, inputs, target_sizes):
        """Handle small API differences between transformers releases."""
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            post_process = self.processor.post_process_grounded_object_detection
            try:
                return post_process(
                    outputs,
                    input_ids=inputs.get("input_ids"),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )
            except TypeError:
                return post_process(
                    outputs,
                    inputs.get("input_ids"),
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=target_sizes,
                )

        if hasattr(self.processor, "post_process_object_detection"):
            return self.processor.post_process_object_detection(
                outputs,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )

        raise AttributeError("processor does not expose a supported object-detection post-processor")


def resolve_model_id(model_size: str, model_id: str | None) -> str:
    if model_id is not None:
        return model_id
    try:
        return MODEL_SIZES[model_size]
    except KeyError as exc:
        raise ValueError(f"model_size must be one of {sorted(MODEL_SIZES)}, got {model_size!r}") from exc


def detect_animals(
    data_root: Path,
    out_dir: Path,
    model_size: str = "tiny",
    model_id: str | None = None,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    presence_threshold: float | None = None,
    prompt: str = "animal.",
    gpu: int = 0,
    output_name: str = "animal_presence.csv",
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Grounding DINO detection requires torch. Install dependencies with "
            "`pip install -r student/requirements.txt`."
        ) from exc

    resolved_model_id = resolve_model_id(model_size, model_id)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    threshold = box_threshold if presence_threshold is None else presence_threshold

    records = discover_images(data_root, splits)
    detector = GroundingDinoAnimalDetector(
        model_id=resolved_model_id,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        prompt=prompt,
    )

    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc="detecting animals"):
        max_confidence = detector.detect_max_confidence(record.path)
        rows.append({
            "uid": record.uid,
            "animal_present": int(max_confidence >= threshold),
            "max_confidence": max_confidence,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_name
    pd.DataFrame(rows, columns=["uid", "animal_present", "max_confidence"]).to_csv(
        output_path,
        index=False,
        float_format="%.6g",
    )
    print(
        f"wrote {output_path} ({len(rows)} rows, model={resolved_model_id}, "
        f"presence_threshold={threshold:g})"
    )
    return output_path


def _find_split_image_path(data_root: Path, split: str, uid: str) -> Path | None:
    """Resolve an image path for a uid under <data_root>/<split>/images/."""
    images_dir = data_root / split / "images"
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = images_dir / f"{uid}{ext}"
        if p.exists():
            return p
    return None


def _load_detection_with_labels(
    data_root: Path,
    detection_csv: Path,
    split: str = "val",
) -> pd.DataFrame | None:
    """Join detection output (uid, animal_present, max_confidence) with labels.csv.

    For val-style labels we define ``has_animal`` as ``y != 0``.
    """
    det = pd.read_csv(detection_csv)
    required_cols = {"uid", "animal_present", "max_confidence"}
    if not required_cols.issubset(det.columns):
        raise ValueError(
            f"{detection_csv} must contain columns {sorted(required_cols)}"
        )

    labels_path = data_root / split / "labels.csv"
    if not labels_path.exists():
        print(f"skipping qualitative plot for {split}: labels not found at {labels_path}")
        return None

    labels = pd.read_csv(labels_path)
    if "uid" not in labels.columns or "y" not in labels.columns:
        raise ValueError(f"{labels_path} must contain at least uid,y columns")

    det = det.copy()
    labels = labels.copy()
    det["uid"] = det["uid"].astype(str)
    labels["uid"] = labels["uid"].astype(str)
    labels["y"] = labels["y"].astype(int)
    if "domain" in labels.columns:
        labels["domain"] = labels["domain"].astype(str)
    else:
        labels["domain"] = "unknown"

    merged = labels[["uid", "y", "domain"]].merge(
        det[["uid", "animal_present", "max_confidence"]],
        on="uid",
        how="left",
        sort=False,
    )

    merged = merged.dropna(subset=["animal_present", "max_confidence"]).copy()
    merged["animal_present"] = merged["animal_present"].astype(int)
    merged["max_confidence"] = merged["max_confidence"].astype(float)
    merged["has_animal"] = (merged["y"] != 0).astype(int)
    return merged


def plot_grounding_dino_examples(
    data_root: Path,
    detection_csv: Path,
    split: str = "val",
    output_path: Path | None = None,
    seed: int = 42,
) -> Path | None:
    """Plot 3 x 10 qualitative panels from Grounding DINO detections.

    Row 1: 10 random images where Grounding DINO predicted animal_present=1.
    Row 2: 10 random false negatives where y != 0 but animal_present=0.
    Row 3: 10 lowest-confidence images among animal_present=1.

    Titles include whether label indicates an animal (y != 0).
    """
    import random

    import matplotlib.pyplot as plt

    df = _load_detection_with_labels(data_root, detection_csv, split=split)
    if df is None:
        return None

    rng = random.Random(seed)

    row1 = df[df["animal_present"] == 1].copy()
    row2 = df[(df["has_animal"] == 1) & (df["animal_present"] == 0)].copy()
    row3 = df[df["animal_present"] == 1].sort_values("max_confidence", ascending=True).copy()

    row1_uids = row1["uid"].tolist()
    rng.shuffle(row1_uids)
    row1_uids = row1_uids[:10]

    row2_uids = row2["uid"].tolist()
    rng.shuffle(row2_uids)
    row2_uids = row2_uids[:10]

    row3_uids = row3["uid"].tolist()[:10]

    row_lookup = df.set_index("uid")
    rows = [
        ("DINO animal_present=1 (random)", row1_uids),
        ("Label has animal (y!=0) but DINO missed", row2_uids),
        ("Lowest max_confidence among DINO animal_present=1", row3_uids),
    ]

    fig, axes = plt.subplots(3, 10, figsize=(42, 14))
    for row_idx, (row_title, uids) in enumerate(rows):
        for col_idx in range(10):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx == 0:
                ax.text(
                    0.02,
                    1.02,
                    row_title,
                    transform=ax.transAxes,
                    fontsize=11,
                    fontweight="bold",
                    va="bottom",
                )

            if col_idx >= len(uids):
                continue

            uid = str(uids[col_idx])
            row = row_lookup.loc[uid]
            img_path = _find_split_image_path(data_root, split, uid)
            if img_path is None:
                ax.text(0.5, 0.5, f"missing\n{uid}", ha="center", va="center")
                continue

            image = Image.open(img_path).convert("RGB")
            ax.imshow(image)
            has_animal_text = "yes" if int(row["has_animal"]) == 1 else "no"
            domain_text = str(row.get("domain", "unknown"))
            ax.set_title(
                f"uid={uid}\ndomain={domain_text} | conf={float(row['max_confidence']):.3f} | label animal={has_animal_text}",
                fontsize=8,
            )

    fig.suptitle("Grounding DINO qualitative checks", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path is None:
        output_path = detection_csv.with_name(f"{detection_csv.stem}_qualitative_grid.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a zero-shot Grounding DINO model to flag images containing any animal."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Path to challenge_data/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the output CSV")
    parser.add_argument("--output-name", type=str, default="animal_presence.csv")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Challenge splits to scan")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="tiny",
                        help="Grounding DINO size alias to use when --model-id is not set")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Explicit Hugging Face model id; overrides --model-size")
    parser.add_argument("--prompt", type=str, default="animal.",
                        help="Zero-shot detection prompt. Keep this generic to avoid species classification.")
    parser.add_argument("--box-threshold", type=float, default=0.25,
                        help="Grounding DINO box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO text match threshold")
    parser.add_argument("--presence-threshold", type=float, default=None,
                        help="Threshold for animal_present; defaults to --box-threshold")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_path = detect_animals(
        data_root=args.data_root,
        out_dir=args.out_dir,
        model_size=args.model_size,
        model_id=args.model_id,
        splits=tuple(args.splits),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        presence_threshold=args.presence_threshold,
        prompt=args.prompt,
        gpu=args.gpu,
        output_name=args.output_name,
    )

    for split in args.splits:
        plot_path = args.out_dir / f"{Path(args.output_name).stem}_{split}_qualitative_grid.png"
        plot_grounding_dino_examples(
            data_root=args.data_root,
            detection_csv=output_path,
            split=split,
            output_path=plot_path,
        )


if __name__ == "__main__":
    main()
