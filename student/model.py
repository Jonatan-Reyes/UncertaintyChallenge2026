"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

The backbone is a frozen, pretrained feature extractor. On top of it sit
several independent small MLP heads, each meant to be trained with its own
IVON optimizer (see ``train.py``) so it learns a posterior over its own
weights instead of a point estimate. Averaging softmax predictions across
heads gives ensemble-style calibrated uncertainty without adapting the
(expensive) backbone per member — the backbone forward pass runs once per
image and is shared by every head.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``    holds the per-member MLP heads (an ``nn.ModuleList``).
- ``self.backbone`` is the frozen feature extractor.
- ``forward(x)``   returns logits of shape ``(N, num_classes)``.
- ``embed(x)``     returns features of shape ``(N, embed_dim)``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

DEFAULT_BACKBONE = "vit_small_patch16_dinov3.lvd1689m"


class Classifier(nn.Module):
    """Frozen timm backbone (feature extractor) + several independent MLP heads.

    The backbone is created via ``timm.create_model(..., num_classes=0)``,
    which returns pooled features rather than logits, and is always frozen
    (``requires_grad=False``) — ``embed`` runs it under ``torch.no_grad()``,
    since the point of this design is that IVON's per-step Monte Carlo
    sampling only has to touch the small heads, not the backbone.

    ``forward`` computes the shared embedding once, runs every head on it,
    and returns the log of the probability-space average across heads (see
    ``forward_members`` for the raw per-head logits, e.g. for ensemble
    disagreement / energy-based uncertainty).
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        num_heads: int = 4,
        head_hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.backbone_name = backbone_name
        self.embed_dim = int(self.backbone.num_features)
        self.num_classes = int(num_classes)
        self.num_members = int(num_heads)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.embed_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, self.num_classes),
            )
            for _ in range(self.num_members)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)``, from the frozen backbone."""
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(x)

    def iter_member_logits(self, x: torch.Tensor):
        """Yield each head's logits, computing the shared backbone embedding once."""
        feats = self.embed(x)
        for head in self.head:
            yield head(feats)

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        """Per-member logits, shape ``(num_members, N, num_classes)``."""
        return torch.stack(list(self.iter_member_logits(x)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log of the probability-space ensemble average, shape ``(N, num_classes)``.

        Averaging each head's softmax (rather than its raw logits) is the
        standard deep-ensemble combination rule and calibrates better.
        Returning ``log(mean_k softmax(logits_k))`` keeps the
        ``forward(x) -> logits`` contract intact: since the averaged
        probabilities already sum to 1, ``softmax(forward(x))`` downstream
        reproduces them exactly, so temperature scaling, ``CrossEntropyLoss``,
        etc. all still work unchanged.
        """
        member_probs = torch.softmax(self.forward_members(x), dim=-1)
        mean_probs = member_probs.mean(dim=0)
        return torch.log(mean_probs.clamp_min(1e-12))
