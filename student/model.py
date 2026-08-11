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

import timm
import torch
import torch.nn as nn

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
        pooling: str | None = None,
        img_size: int | None = None,
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
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        # 'cls' / 'avg' pool the raw token sequence ourselves instead of
        # relying on timm's global_pool: some ViT checkpoints (e.g. DINOv2)
        # only ship a `norm` layer and don't have the separate `fc_norm`
        # timm's 'avg' head expects, so overriding global_pool at
        # construction time breaks strict state-dict loading.
        self.pooling = pooling

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample feature vectors, shape ``(N, embed_dim)``."""
        if self.pooling is None:
            return self.backbone(x)
        tokens = self.backbone.forward_features(x)
        if self.pooling == "cls":
            return tokens[:, 0]
        if self.pooling == "avg":
            return tokens[:, self.backbone.num_prefix_tokens:].mean(dim=1)
        raise ValueError(f"unknown pooling {self.pooling!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
