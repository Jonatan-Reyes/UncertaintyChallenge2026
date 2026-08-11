"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

Things to experiment with:
- Swap the backbone via ``backbone_name`` (any timm model id, e.g.
  ``"resnet18"``, ``"convnext_small"``, ``"vit_base_patch16_224"``).
- Replace ``embed`` with a custom feature extractor / pooling scheme.
- Add a projection head between the backbone and the classifier.
- Enable dropout at eval time and override ``forward`` to average MC samples.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``  holds the classifier params; they get the higher LR.
- ``self.backbone`` is everything else; its params get the lower LR.
- ``forward(x)`` returns logits of shape ``(N, num_classes)``.
- ``embed(x)``   returns features of shape ``(N, embed_dim)``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

DEFAULT_BACKBONE = "vit_small_patch16_dinov3.lvd1689m"

# Linear submodule names LoRA is injected into (ViT attention qkv/proj + MLP fc1/fc2).
LORA_TARGET_MODULES = ("qkv", "proj", "fc1", "fc2")


class Classifier(nn.Module):
    """timm backbone (as feature extractor) + an ensemble of LoRA-adapted heads.

    The backbone is created via ``timm.create_model(..., num_classes=0)``,
    which returns pooled features rather than logits — no ``nn.Identity``
    plumbing needed.

    Set ``pretrained=True`` to use timm's published pretrained weights
    (recommended for real training; off by default so the test suite stays
    offline).

    With ``lora_r > 0`` (default), the backbone is frozen and given
    ``num_lora_members`` independent LoRA adapters (via ``peft``, one per
    ensemble member) on every ``nn.Linear`` named in ``LORA_TARGET_MODULES``,
    each paired with its own linear head in ``self.head``. ``forward`` runs
    one pass per member (switching the active adapter each time) and returns
    the mean logits — the standard classification-loss contract, so
    ``train``/``eval``/``predict`` don't need to change. Use
    ``forward_members`` to get the K individual predictions for
    ensemble-uncertainty estimates (e.g. predictive variance).

    ``lora_r=0`` disables LoRA and falls back to a single directly
    fine-tuned backbone with one head.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        num_lora_members: int = 4,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.backbone_name = backbone_name
        self.embed_dim = int(self.backbone.num_features)
        self.num_classes = int(num_classes)
        self.lora_r = lora_r
        self.num_members = num_lora_members if lora_r > 0 else 1

        if lora_r > 0:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(LORA_TARGET_MODULES),
            )
            self.backbone = get_peft_model(self.backbone, lora_config, adapter_name="lora_0")
            for i in range(1, self.num_members):
                self.backbone.add_adapter(f"lora_{i}", lora_config)
            self.adapter_names = [f"lora_{i}" for i in range(self.num_members)]
        else:
            self.adapter_names = None

        self.head = nn.ModuleList(
            nn.Linear(self.embed_dim, self.num_classes) for _ in range(self.num_members)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)``, from the active adapter."""
        return self.backbone(x)

    def iter_member_logits(self, x: torch.Tensor):
        """Yield each member's logits one at a time.

        Lets a caller ``backward()`` per member before moving to the next, so
        only one member's activations are ever resident in memory instead of
        all ``num_members`` at once.
        """
        for i, head in enumerate(self.head):
            if self.adapter_names is not None:
                self.backbone.set_adapter(self.adapter_names[i])
            yield head(self.embed(x))

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        """Per-member logits, shape ``(num_members, N, num_classes)``."""
        return torch.stack(list(self.iter_member_logits(x)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=0)
