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

    detect_animals(
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


if __name__ == "__main__":
    main()
