"""PyTorch Multimodal Product Classifier Architecture.
Combines a pre-trained CNN (ResNet or MobileNet) for image representations
with an embedding or TF-IDF layer for text representations, fused via
concatenation into a classification head.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models


def build_image_backbone(
    backbone_name: str = "resnet18",
    pretrained: bool = True,
    freeze: bool = False,
) -> Tuple[nn.Module, int]:
    """Instantiate pre-trained image feature extractor and extract pooled feature dimension.

    Replaces the final classification layer with nn.Identity().
    """
    backbone_name = backbone_name.lower().strip()

    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        img_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()

    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        img_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()

    elif backbone_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3 small classifier[0] in_features is 576
        img_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

    elif backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        img_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. "
            "Supported: 'resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v2'"
        )

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, img_dim


class EmbeddingBagTextEncoder(nn.Module):
    """Encodes token IDs into a dense fixed-length text vector using mean-pooled embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        mode: str = "mean",
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            mode=mode,
            padding_idx=padding_idx,
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        tokens: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            tokens: 1D flat tensor if offsets provided; or 2D (B, L) padded
            tensor.
            offsets: 1D tensor marking starting indices of each sample in
            tokens.

        Returns:
            Tensor of shape (B, embed_dim)
        """
        if offsets is not None:
            return self.embedding_bag(tokens, offsets)
        return self.embedding_bag(tokens)


class TFIDFTextEncoder(nn.Module):
    """Encodes TF-IDF vector into dense feature space via a linear projection."""

    def __init__(self, in_features: int, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.ReLU(),
        )
        self.embed_dim = embed_dim

    def forward(self, tfidf_tensor: torch.Tensor) -> torch.Tensor:
        return self.proj(tfidf_tensor)


class MultimodalProductClassifier(nn.Module):
    """Multimodal Classifier combining Image CNN and Text Embeddings via feature concatenation.

    Supports optional modality ablation for evaluating modality-specific lift.
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        image_backbone: str = "resnet18",
        pretrained_image: bool = True,
        freeze_image: bool = False,
        text_encoder_type: str = "embedding_bag",
        text_embed_dim: int = 128,
        tfidf_dim: Optional[int] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.text_encoder_type = text_encoder_type

        # 1. Image Branch
        self.image_encoder, self.img_dim = build_image_backbone(
            backbone_name=image_backbone,
            pretrained=pretrained_image,
            freeze=freeze_image,
        )

        # 2. Text Branch
        if text_encoder_type == "embedding_bag":
            self.text_encoder = EmbeddingBagTextEncoder(
                vocab_size=vocab_size,
                embed_dim=text_embed_dim,
            )
            self.text_dim = text_embed_dim
        elif text_encoder_type == "tfidf":
            if tfidf_dim is None:
                raise ValueError("tfidf_dim must be provided when text_encoder_type is 'tfidf'")
            self.text_encoder = TFIDFTextEncoder(
                in_features=tfidf_dim,
                embed_dim=text_embed_dim,
            )
            self.text_dim = text_embed_dim
        else:
            raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")

        # 3. Multimodal Fusion & Head
        fused_dim = self.img_dim + self.text_dim
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(fused_dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(fused_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector from the image branch."""
        return self.image_encoder(images)

    def extract_text_features(
        self,
        tokens: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        tfidf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract text representation vector."""
        if self.text_encoder_type == "embedding_bag":
            return self.text_encoder(tokens, offsets)
        elif self.text_encoder_type == "tfidf":
            return self.text_encoder(tfidf)
        raise RuntimeError("Invalid text encoder configuration")

    def forward(
        self,
        images: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        tfidf: Optional[torch.Tensor] = None,
        ablate_image: bool = False,
        ablate_text: bool = False,
    ) -> torch.Tensor:
        """Forward pass concatenating image and text features into the classification head.

        Args:
            images: Tensor of shape (B, 3, H, W)
            tokens: Token tensor for text encoder (flat or 2D)
            offsets: Offset tensor for EmbeddingBag if flat tokens used
            tfidf: TF-IDF feature tensor if tfidf encoder used
            ablate_image: If True, zeros out the image features (text-only evaluation)
            ablate_text: If True, zeros out the text features (image-only evaluation)

        Returns:
            Class logits of shape (B, num_classes)
        """
        batch_size = images.size(0)

        # Image representation
        if ablate_image:
            img_feat = torch.zeros(
                batch_size, self.img_dim, device=images.device, dtype=images.dtype
            )
        else:
            img_feat = self.extract_image_features(images)

        # Text representation
        if ablate_text:
            text_feat = torch.zeros(
                batch_size, self.text_dim, device=images.device, dtype=images.dtype
            )
        else:
            text_feat = self.extract_text_features(tokens=tokens, offsets=offsets, tfidf=tfidf)

        # Late Fusion via Concatenation
        fused = torch.cat([img_feat, text_feat], dim=-1)

        # Final classification logits
        logits = self.classifier(fused)
        return logits
