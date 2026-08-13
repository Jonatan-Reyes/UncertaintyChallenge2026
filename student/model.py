"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

An ensemble of 5 different pretrained foundation vision backbones (not 5
copies of the same one) — each gets its own linear head, and predictions
are combined by averaging softmax probabilities across the 5 models.
Diversity comes from the backbones themselves (different architectures /
pretraining objectives), not from LoRA adapters or posterior sampling. Most
of each backbone is frozen; only its last 2 layers are fine-tuned alongside
the head.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``      per-backbone linear heads (an ``nn.ModuleList``).
- ``self.backbones``  the (mostly frozen) feature extractors (an ``nn.ModuleList``).
- ``forward(x)``      returns logits of shape ``(N, num_classes)``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# 5 different foundation backbones: 2 self-supervised ViTs (DINOv3, DINOv2),
# 1 vision-language contrastive model (SigLIP), 1 CNN (ConvNeXt), 1 masked-
# image-modeling ViT (EVA-02) -- deliberately different architectures /
# pretraining objectives, not 5 seeds of the same model.
DEFAULT_BACKBONES = [
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_siglip_224.webli",
    "convnext_tiny.fb_in22k",
    "eva02_small_patch14_224.mim_in22k",
]


def _unfreeze_last_n_layers(backbone: nn.Module, n: int = 2) -> None:
    """Unfreeze the last ``n`` layers of a timm backbone, plus its final
    norm if present.

    Looks for the common ``.blocks`` (ViT-style: DINOv3/DINOv2/SigLIP/EVA-02)
    or ``.stages`` (ConvNeXt-style) attribute for a reasonable notion of
    "layer" across architectures; falls back to the backbone's last ``n``
    top-level children otherwise.
    """
    if hasattr(backbone, "blocks"):
        layers = list(backbone.blocks)
    elif hasattr(backbone, "stages"):
        layers = list(backbone.stages)
    else:
        layers = list(backbone.children())

    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True

    if hasattr(backbone, "norm") and isinstance(backbone.norm, nn.Module):
        for p in backbone.norm.parameters():
            p.requires_grad = True


class BrierLoss(nn.Module):
    """Brier score as a differentiable loss: mean_i sum_k (p_ik - onehot_ik)^2.

    Unlike ECE, Brier is already a smooth quadratic function of the
    probabilities, so this matches ``metrics.brier()`` exactly rather than
    needing a soft approximation — usable directly as an LBFGS objective
    (e.g. for temperature scaling).
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        return ((probs - onehot) ** 2).sum(dim=1).mean()


class Classifier(nn.Module):
    """Ensemble of foundation backbones, each with its own linear head.

    Every backbone is created via ``timm.create_model(..., num_classes=0)``
    (pooled features). All backbone parameters start frozen
    (``requires_grad=False``), then ``_unfreeze_last_n_layers`` re-enables
    gradients on just the last 2 layers of each — the rest stays fixed.
    Backbones have different ``embed_dim``s, so each gets its own
    appropriately-sized head rather than sharing one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_names: list[str] | None = None,
        pretrained: bool = False,
        num_unfrozen_layers: int = 2,
    ):
        super().__init__()
        self.backbone_names = list(backbone_names) if backbone_names else list(DEFAULT_BACKBONES)
        self.num_classes = int(num_classes)
        self.num_members = len(self.backbone_names)

        self.backbones = nn.ModuleList()
        self.head = nn.ModuleList()
        for name in self.backbone_names:
            bb = timm.create_model(name, pretrained=pretrained, num_classes=0)
            for p in bb.parameters():
                p.requires_grad = False
            _unfreeze_last_n_layers(bb, n=num_unfrozen_layers)
            self.backbones.append(bb)
            self.head.append(nn.Linear(int(bb.num_features), self.num_classes))

    def iter_member_logits(self, x: torch.Tensor):
        """Yield each backbone+head's logits one at a time.

        No ``torch.no_grad()`` here — unlike a fully-frozen backbone, the
        last few unfrozen layers need gradients to flow through.
        """
        for bb, head in zip(self.backbones, self.head):
            feat = bb(x)
            yield head(feat)

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        """Per-member logits, shape ``(num_members, N, num_classes)``."""
        return torch.stack(list(self.iter_member_logits(x)), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log of the probability-space ensemble average, shape ``(N, num_classes)``.

        Averaging each member's softmax (rather than its raw logits) is the
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
