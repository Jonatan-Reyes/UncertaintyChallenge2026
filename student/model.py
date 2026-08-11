"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

Things to experiment with:
- Swap the backbone via ``backbone_name`` (any timm model id, e.g.
  ``"resnet18"``, ``"convnext_small"``, ``"vit_base_patch16_224"``).
- Replace ``embed`` with a custom feature extractor / pooling scheme.
- Add a projection head between the backbone and the classifier.
- Enable dropout at eval time and override ``forward`` to average MC samples.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``  is the linear classifier; its parameters get the higher LR.
- ``self.backbone`` is everything else; its parameters get the lower LR.
- ``forward(x)`` returns logits of shape ``(N, num_classes)``.
- ``embed(x)``   returns features of shape ``(N, embed_dim)``.
"""

from __future__ import annotations

from os import PathLike

import timm
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # pragma: no cover - only needed when LoRA is enabled
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

DEFAULT_BACKBONE = "convnext_tiny"


class Classifier(nn.Module):
    """timm backbone (as feature extractor) + linear classifier head.

    The backbone is created via ``timm.create_model(..., num_classes=0)``,
    which returns pooled features rather than logits — no ``nn.Identity``
    plumbing needed. ``self.head`` is the new ``num_classes``-way classifier.

    Set ``pretrained=True`` to use timm's published pretrained weights
    (recommended for real training; off by default so the test suite stays
    offline).
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.backbone_name = backbone_name
        self.embed_dim = int(self.backbone.num_features)
        self.num_classes = int(num_classes)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)``."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class LoRAClassifier(Classifier):
    """Classifier variant that injects PEFT LoRA adapters into the backbone."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
        modules_to_save: tuple[str, ...] = ("head",),
    ):
        super().__init__(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=pretrained,
        )
        if LoraConfig is None or get_peft_model is None or PeftModel is None:
            raise ImportError(
                "peft is required when LoRA is enabled. Install with: pip install peft"
            )
        if lora_r <= 0:
            raise ValueError("lora_r must be > 0 for LoRAClassifier")

        self.lora_config: dict[str, object] = {
            "r": int(lora_r),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "target_modules": tuple(lora_target_modules),
            "modules_to_save": tuple(modules_to_save),
        }

        peft_cfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            target_modules=list(lora_target_modules),
            lora_dropout=float(lora_dropout),
            modules_to_save=list(modules_to_save),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)

    def save_pretrained(self, save_directory: str | PathLike[str]) -> None:
        self.backbone.save_pretrained(save_directory)

    def load_pretrained(self, load_directory: str | PathLike[str], is_trainable: bool = False) -> None:
        self.backbone = PeftModel.from_pretrained(
            self.backbone,
            load_directory,
            is_trainable=is_trainable,
        )
