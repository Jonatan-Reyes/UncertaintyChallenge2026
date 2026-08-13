"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

An ensemble of 5 different pretrained foundation vision backbones (not 5
copies of the same one) — each gets 2 linear heads, both trained on
cross-entropy plus an alpha-weighted secondary calibration term (one head
adds Brier, the other adds ECE — see ``train.py``'s ``Trainer.criteria`` and
``CombinedLoss`` below), and predictions are combined by averaging softmax
probabilities across all 10 members. Diversity comes both from the
backbones themselves (different architectures / pretraining objectives)
and from each backbone's 2 heads optimizing different loss mixes. Most of
each backbone is frozen; only its last 2 layers are fine-tuned alongside
its heads.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``      per-backbone linear heads (an ``nn.ModuleList``).
- ``self.backbones``  the (mostly frozen) feature extractors (an ``nn.ModuleList``).
- ``forward(x)``      returns logits of shape ``(N, num_classes)``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from student.metrics import ECE_BINS

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

    for layer in (layers[-n:] if n > 0 else []):
        for p in layer.parameters():
            p.requires_grad = True

    if n > 0 and hasattr(backbone, "norm") and isinstance(backbone.norm, nn.Module):
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


class SoftECELoss(nn.Module):
    """Differentiable ECE surrogate: soft (sigmoid) bin membership instead of
    the hard ``>=``/``<`` comparisons in ``metrics.ece``, so gradients can
    flow through it — true ECE is piecewise-constant (~zero gradient
    everywhere) and can't be used as a training loss directly.
    """

    def __init__(self, n_bins: int = ECE_BINS, sharpness: float = 50.0):
        super().__init__()
        self.register_buffer("edges", torch.linspace(0.0, 1.0, n_bins + 1))
        self.sharpness = sharpness

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        correct = (preds == labels).float()
        n = confs.shape[0]
        loss = confs.new_zeros(())
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            w = torch.sigmoid((confs - lo) * self.sharpness) - torch.sigmoid((confs - hi) * self.sharpness)
            wsum = w.sum() + 1e-12
            bin_conf = (w * confs).sum() / wsum
            bin_acc = (w * correct).sum() / wsum
            loss = loss + torch.abs(bin_conf - bin_acc) * (wsum / n)
        return loss


class CombinedLoss(nn.Module):
    """``primary(logits, labels) + alpha * secondary(logits, labels)``."""

    def __init__(self, primary: nn.Module, secondary: nn.Module, alpha: float):
        super().__init__()
        self.primary = primary
        self.secondary = secondary
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.primary(logits, labels) + self.alpha * self.secondary(logits, labels)


class Classifier(nn.Module):
    """Ensemble of foundation backbones, each with ``heads_per_backbone``
    linear heads (default 2 — see ``train.py``'s ``Trainer.criteria``, built
    from ``CombinedLoss``: cross-entropy plus an alpha-weighted secondary
    calibration term, one head per secondary loss).

    Every backbone is created via ``timm.create_model(..., num_classes=0)``
    (pooled features). All backbone parameters start frozen
    (``requires_grad=False``), then ``_unfreeze_last_n_layers`` re-enables
    gradients on just the last 2 layers of each — the rest stays fixed.
    Backbones have different ``embed_dim``s, so each gets its own
    appropriately-sized heads rather than sharing one. Each backbone's
    embedding is computed once and shared across its own heads.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_names: list[str] | None = None,
        pretrained: bool = False,
        num_unfrozen_layers: int = 2,
        heads_per_backbone: int = 2,
    ):
        super().__init__()
        self.backbone_names = list(backbone_names) if backbone_names else list(DEFAULT_BACKBONES)
        self.num_classes = int(num_classes)
        self.heads_per_backbone = int(heads_per_backbone)

        self.backbones = nn.ModuleList()
        self.head = nn.ModuleList()
        for name in self.backbone_names:
            # ViT-family backbones (vit_*, eva*) hard-assert input resolution
            # against their registered default (often not 224, e.g. DINOv2's
            # is 518) unless created with dynamic_img_size=True, which lets
            # them interpolate position embeddings for other sizes. ConvNeXt
            # and other non-ViT architectures don't accept that kwarg.
            extra_kwargs = {"dynamic_img_size": True} if name.startswith(("vit_", "eva")) else {}
            if name.startswith("eva"):
                # Mean-pool over patch tokens instead of the class token.
                extra_kwargs["global_pool"] = "avg"
            bb = timm.create_model(name, pretrained=pretrained, num_classes=0, **extra_kwargs)
            for p in bb.parameters():
                p.requires_grad = False
            _unfreeze_last_n_layers(bb, n=num_unfrozen_layers)
            self.backbones.append(bb)
            for _ in range(self.heads_per_backbone):
                self.head.append(nn.Linear(int(bb.num_features), self.num_classes))

        self.num_members = len(self.head)

    def iter_member_logits(self, x: torch.Tensor):
        """Yield each backbone's heads' logits one at a time, in
        ``[backbone_0_head_0, backbone_0_head_1, ..., backbone_1_head_0, ...]``
        order — each backbone's embedding is computed once and reused by its
        own ``heads_per_backbone`` heads.

        No ``torch.no_grad()`` here — unlike a fully-frozen backbone, the
        last few unfrozen layers need gradients to flow through.
        """
        idx = 0
        for bb in self.backbones:
            feat = bb(x)
            for _ in range(self.heads_per_backbone):
                yield self.head[idx](feat)
                idx += 1

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        """Per-member logits, shape ``(num_members, N, num_classes)``."""
        return torch.stack(list(self.iter_member_logits(x)), dim=0)

    def iter_backbone_logits(self, x: torch.Tensor):
        """Yield each backbone's own combined logits (log of that backbone's
        heads' probability-space average) one backbone at a time — the
        per-backbone analogue of ``iter_member_logits``, used for fitting an
        independent temperature per backbone.
        """
        idx = 0
        for bb in self.backbones:
            feat = bb(x)
            head_logits = torch.stack(
                [self.head[idx + j](feat) for j in range(self.heads_per_backbone)], dim=0
            )
            idx += self.heads_per_backbone
            probs = torch.softmax(head_logits, dim=-1).mean(dim=0)
            yield torch.log(probs.clamp_min(1e-12))

    def forward_per_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Per-backbone combined logits, shape ``(num_backbones, N, num_classes)``."""
        return torch.stack(list(self.iter_backbone_logits(x)), dim=0)

    def forward(self, x: torch.Tensor, temperatures: list[float] | None = None) -> torch.Tensor:
        """Log of the probability-space ensemble average across backbones,
        shape ``(N, num_classes)``.

        Averaging each member's softmax (rather than its raw logits) is the
        standard deep-ensemble combination rule and calibrates better.
        Returning ``log(mean_k softmax(logits_k))`` keeps the
        ``forward(x) -> logits`` contract intact: since the averaged
        probabilities already sum to 1, ``softmax(forward(x))`` downstream
        reproduces them exactly, so temperature scaling, ``CrossEntropyLoss``,
        etc. all still work unchanged.

        If ``temperatures`` is given (one scalar per backbone), each
        backbone's own combined distribution is temperature-scaled
        *independently* before joining the cross-backbone average — rather
        than one shared temperature applied after combining everything.
        """
        backbone_logits = self.forward_per_backbone(x)  # (num_backbones, N, C)
        if temperatures is not None:
            t = torch.tensor(temperatures, device=backbone_logits.device, dtype=backbone_logits.dtype)
            backbone_logits = backbone_logits / t.view(-1, 1, 1)
        member_probs = torch.softmax(backbone_logits, dim=-1)
        mean_probs = member_probs.mean(dim=0)
        return torch.log(mean_probs.clamp_min(1e-12))
