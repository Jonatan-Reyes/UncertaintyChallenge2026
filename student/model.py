"""Model definitions for the challenge.

Defines the ``Classifier`` nn.Module used by ``train``, ``eval``, and ``predict``.

The minimal contract (so train / eval / predict don't need to change):

- ``self.head``  is the classifier (``nn.Linear``, or ``nn.ModuleList`` of
  ``num_heads`` linears); its parameters get the higher LR.
- ``self.backbone`` is everything else; its parameters get the lower LR.
- ``forward(x)`` returns logits of shape ``(N, num_classes)`` — the mean
  over heads when ``num_heads > 1``.
- ``embed(x)``   returns pooled features of shape ``(N, embed_dim)``.

Things to experiment with:
- Swap the backbone via ``backbone_name`` (any timm model id).
- ``pooling``: cls / avg / cls_avg / gem / attn — see ``embed``.
- ``lora``: dict(r=8, alpha=16, target_blocks=8) to fine-tune the last N
  blocks' attention qkv/proj via LoRA instead of a frozen backbone.
- ``num_heads``: K independently-initialized linear heads for a cheap
  within-backbone ensemble + disagreement signal (``forward_heads``).
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_BACKBONE = "convnext_tiny"
POOLING_MODES = ("cls", "avg", "cls_avg", "gem", "attn")


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a low-rank additive update.

    ``y = W0 x + b0 + (alpha / r) * B(A x)``. ``W0``/``b0`` stay frozen (the
    pretrained path is untouched); only ``A``, ``B`` train. ``B`` is
    zero-initialized so the wrapped module is an exact identity at step 0.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return out + update


def _apply_lora(backbone: nn.Module, r: int, alpha: int, target_blocks: int) -> int:
    """Replace ``qkv``/``proj``/``fc1``/``fc2`` Linear layers in the last
    ``target_blocks`` transformer blocks with ``LoRALinear`` wrappers.
    Freezes every other backbone parameter. Returns the number of blocks
    actually touched (backbones expose blocks as ``.blocks``).
    """
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError(f"backbone {backbone.__class__.__name__} has no `.blocks` — LoRA needs a ViT-style model")
    for p in backbone.parameters():
        p.requires_grad_(False)

    target_names = ("qkv", "proj", "fc1", "fc2")
    touched = list(blocks)[-target_blocks:]
    for block in touched:
        for parent_name, parent in block.named_modules():
            for attr in list(parent._modules.keys()):
                if attr in target_names and isinstance(parent._modules[attr], nn.Linear):
                    parent._modules[attr] = LoRALinear(parent._modules[attr], r=r, alpha=alpha)
    return len(touched)


class GeM(nn.Module):
    """Generalized-mean pooling over patch tokens, learnable exponent p."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), float(p)))
        self.eps = eps

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=1).pow(1.0 / self.p)


class AttentionPool(nn.Module):
    """Single learned query attending over patch tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * dim ** -0.5)
        self.scale = dim ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(tokens.size(0), -1, -1)
        attn = (q @ tokens.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return (attn @ tokens).squeeze(1)


class Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = DEFAULT_BACKBONE,
        pretrained: bool = False,
        pooling: str | None = None,
        img_size: int | None = None,
        num_heads: int = 1,
        head_dropout: float = 0.0,
        lora: dict | None = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        backbone_kwargs = {}
        if img_size is not None:
            backbone_kwargs["img_size"] = img_size
        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, **backbone_kwargs
            )
        except TypeError:
            # fully-convolutional backbones (e.g. ConvNeXt) don't take img_size
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.backbone_name = backbone_name
        self.embed_dim = int(self.backbone.num_features)
        self.num_classes = int(num_classes)

        if pooling is not None and pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling {pooling!r}, choose from {POOLING_MODES}")
        self.pooling = pooling
        pool_out_dim = self.embed_dim * 2 if pooling == "cls_avg" else self.embed_dim
        self.gem = GeM() if pooling == "gem" else None
        self.attn_pool = AttentionPool(self.embed_dim) if pooling == "attn" else None

        self.num_heads = int(num_heads)
        self.head_dropout = float(head_dropout)
        if self.num_heads == 1:
            self.head = nn.Linear(pool_out_dim, self.num_classes)
        else:
            self.head = nn.ModuleList(
                nn.Linear(pool_out_dim, self.num_classes) for _ in range(self.num_heads)
            )

        # DDP training needs per-head logits (N,K,C) to reach the caller through
        # this same forward() -- calling forward_heads() directly on a DDP-wrapped
        # module bypasses DDP.__call__'s gradient-sync hooks (see train_ddp.py).
        # Toggling this flag around the forward call keeps eval/predict's default
        # (mean-over-heads) contract untouched.
        self._return_all_heads = False

        self.lora_cfg = lora
        if lora is not None:
            _apply_lora(self.backbone, r=lora.get("r", 8), alpha=lora.get("alpha", 16),
                       target_blocks=lora.get("target_blocks", 8))

        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)`` (``2*embed_dim``
        for ``cls_avg``)."""
        if self.pooling is None:
            return self.backbone(x)
        tokens = self.backbone.forward_features(x)
        n_prefix = self.backbone.num_prefix_tokens
        if self.pooling == "cls":
            return tokens[:, 0]
        if self.pooling == "avg":
            return tokens[:, n_prefix:].mean(dim=1)
        if self.pooling == "cls_avg":
            return torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=1)
        if self.pooling == "gem":
            return self.gem(tokens[:, n_prefix:])
        if self.pooling == "attn":
            return self.attn_pool(tokens[:, n_prefix:])
        raise ValueError(f"unknown pooling {self.pooling!r}")

    def forward_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head logits, shape ``(N, num_heads, num_classes)``.

        Each head sees its own independent dropout mask over the pooled
        features (``F.dropout`` draws a fresh mask per call). On a frozen
        backbone, a linear head + softmax + CE is a convex problem, so
        different random inits alone tend to converge toward the same
        optimum -- forcing each head to fit a different random feature
        subset is what actually keeps them from collapsing to duplicates.
        No-op at eval time (``training=self.training`` -> False).
        """
        feats = self.embed(x)
        if self.num_heads == 1:
            return self.head(feats).unsqueeze(1)
        outs = []
        for h in self.head:
            f = F.dropout(feats, p=self.head_dropout, training=self.training) if self.head_dropout > 0 else feats
            outs.append(h(f))
        return torch.stack(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_heads == 1:
            return self.head(self.embed(x))
        out = self.forward_heads(x)
        return out if self._return_all_heads else out.mean(dim=1)
