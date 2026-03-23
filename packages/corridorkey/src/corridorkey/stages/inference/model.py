"""Neural network architecture for the CorridorKey chroma keying model.

Defines GreenFormer, a hybrid transformer and CNN model built on a Hiera
encoder with two SegFormer-style decoder heads (alpha and foreground) and
an optional dilated CNN refiner stage.

Migrated from corridorkey-core/model_transformer.py into corridorkey
so that corridorkey is fully self-contained.
"""

from __future__ import annotations

import logging
from typing import Protocol, cast

import timm
import torch
import torch.nn as nn
from torch.nn import functional

logger = logging.getLogger(__name__)


class _PatchEmbedContainer(Protocol):
    proj: nn.Conv2d


class MLP(nn.Module):
    """Linear projection from one channel dimension to another.

    Used in DecoderHead to unify feature map channel counts before fusion.

    Args:
        input_dim: Number of input channels.
        embed_dim: Number of output channels.
    """

    def __init__(self, input_dim: int = 2048, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DecoderHead(nn.Module):
    """Multi-scale feature fusion decoder based on the SegFormer MLP decoder.

    Takes four feature maps from the encoder at different scales, projects each
    to a common embedding dimension, upsamples them to the same spatial size,
    concatenates, and predicts a dense output map.

    Args:
        feature_channels: Channel counts for the four encoder feature levels
            [c1, c2, c3, c4]. Defaults to Hiera Base Plus values [112, 224, 448, 896].
        embedding_dim: Shared channel dimension after per-level projection.
        output_dim: Number of output channels (1 for alpha, 3 for foreground).
    """

    def __init__(
        self, feature_channels: list[int] | None = None, embedding_dim: int = 256, output_dim: int = 1
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        self.linear_c4 = MLP(input_dim=feature_channels[3], embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=feature_channels[2], embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=feature_channels[1], embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=feature_channels[0], embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Project, upsample, fuse, and classify the encoder feature maps.

        Args:
            features: List of four tensors [c1, c2, c3, c4] with shapes
                [B, C_i, H_i, W_i] at decreasing spatial resolutions.

        Returns:
            Dense prediction tensor with shape [B, output_dim, H/4, W/4].
        """
        c1, c2, c3, c4 = features
        batch_size, _, _, _ = c4.shape

        proj_c4 = (
            self.linear_c4(c4.flatten(2).transpose(1, 2)).transpose(1, 2).view(batch_size, -1, c4.shape[2], c4.shape[3])
        )
        proj_c4 = functional.interpolate(proj_c4, size=c1.shape[2:], mode="bilinear", align_corners=False)

        proj_c3 = (
            self.linear_c3(c3.flatten(2).transpose(1, 2)).transpose(1, 2).view(batch_size, -1, c3.shape[2], c3.shape[3])
        )
        proj_c3 = functional.interpolate(proj_c3, size=c1.shape[2:], mode="bilinear", align_corners=False)

        proj_c2 = (
            self.linear_c2(c2.flatten(2).transpose(1, 2)).transpose(1, 2).view(batch_size, -1, c2.shape[2], c2.shape[3])
        )
        proj_c2 = functional.interpolate(proj_c2, size=c1.shape[2:], mode="bilinear", align_corners=False)

        proj_c1 = (
            self.linear_c1(c1.flatten(2).transpose(1, 2)).transpose(1, 2).view(batch_size, -1, c1.shape[2], c1.shape[3])
        )

        fused_features = self.linear_fuse(torch.cat([proj_c4, proj_c3, proj_c2, proj_c1], dim=1))
        # BatchNorm2d does not support bf16; upcast to match BN weight dtype then restore.
        bn_dtype = next(self.bn.parameters()).dtype
        fused_features = self.bn(fused_features.to(bn_dtype)).to(proj_c1.dtype)
        fused_features = self.relu(fused_features)

        x = self.dropout(fused_features)
        x = self.classifier(x)

        return x


class RefinerBlock(nn.Module):
    """Dilated residual block with GroupNorm.

    Expands the receptive field without increasing spatial resolution or
    parameter count. Safe for batch size 1 due to GroupNorm instead of BatchNorm.

    Args:
        channels: Number of input and output channels.
        dilation: Dilation factor for both convolutions.
    """

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out


class CNNRefinerModule(nn.Module):
    """Dilated residual CNN that corrects coarse decoder predictions.

    Stacks four RefinerBlocks with dilation rates 1, 2, 4, 8 giving a
    receptive field of approximately 65 pixels. Designed to fix macroblocking
    artifacts produced by the Hiera transformer backbone.

    The refiner outputs delta logits that are added residually to the coarse
    decoder logits before the final sigmoid activation.

    Args:
        in_channels: Input channels (RGB + coarse predictions = 3 + 4 = 7).
        hidden_channels: Internal channel width throughout the residual blocks.
        out_channels: Output channels matching the coarse prediction count (4).
    """

    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)

        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

        nn.init.normal_(self.final.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.final.bias, 0)  # type: ignore[arg-type]

    def forward(self, img: torch.Tensor, coarse_pred: torch.Tensor) -> torch.Tensor:
        """Predict delta logits from the RGB image and coarse predictions.

        Args:
            img: RGB image tensor with shape [B, 3, H, W].
            coarse_pred: Concatenated coarse alpha and foreground probabilities
                with shape [B, 4, H, W].

        Returns:
            Delta logit tensor with shape [B, 4, H, W], scaled by 10x to allow
            small network outputs to produce meaningful corrections.
        """
        x = torch.cat([img, coarse_pred], dim=1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * 10.0


class GreenFormer(nn.Module):
    """Transformer-based chroma keying model with a CNN refinement stage.

    Combines a Hiera encoder with two SegFormer-style decoder heads (alpha and
    foreground) and an optional dilated CNN refiner. The encoder input is patched
    to accept 4 channels (RGB + mask) instead of the standard 3.

    Args:
        encoder_name: timm model name for the Hiera backbone.
        in_channels: Number of input channels. Patch embedding is extended when
            this differs from 3.
        img_size: Square spatial resolution the model operates at.
        use_refiner: Whether to attach the CNNRefinerModule.
        embedding_dim: Shared embedding dimension for the decoder heads.
    """

    def __init__(
        self,
        encoder_name: str = "hiera_base_plus_224.mae_in1k_ft_in1k",
        in_channels: int = 4,
        img_size: int = 512,
        use_refiner: bool = True,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()

        logger.info("Initializing %s (img_size=%d)", encoder_name, img_size)
        self.encoder = timm.create_model(encoder_name, pretrained=False, features_only=True, img_size=img_size)
        logger.info("Skipped downloading base weights (relying on custom checkpoint)")

        if in_channels != 3:
            self._patch_input_layer(in_channels)

        try:
            feature_channels = self.encoder.feature_info.channels()  # type: ignore[union-attr]
        except (AttributeError, TypeError):
            feature_channels = [112, 224, 448, 896]
        logger.info("Feature channels: %s", feature_channels)

        self.alpha_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=1)
        self.fg_decoder = DecoderHead(feature_channels, embedding_dim, output_dim=3)

        self.use_refiner = use_refiner
        if self.use_refiner:
            self.refiner = CNNRefinerModule(in_channels=7, hidden_channels=64, out_channels=4)
        else:
            self.refiner = None
            logger.info("Refiner module DISABLED (backbone-only mode)")

    def _patch_input_layer(self, in_channels: int) -> None:
        """Extend the patch embedding convolution to accept more than 3 input channels.

        Copies the existing RGB weights into the first 3 input channels of the new
        convolution and initializes any additional channels to zero.

        Args:
            in_channels: Total number of input channels after patching.
        """
        model_obj = getattr(self.encoder, "model", None)
        patch_embed_obj = getattr(model_obj, "patch_embed", None) if model_obj is not None else None
        if patch_embed_obj is None:
            patch_embed_obj = getattr(self.encoder, "patch_embed", None)
        if patch_embed_obj is None:
            raise AttributeError("Could not locate patch_embed module on encoder")

        patch_embed_container = cast(_PatchEmbedContainer, patch_embed_obj)
        patch_embed = patch_embed_container.proj
        weight = patch_embed.weight.detach().clone()
        bias = patch_embed.bias.detach().clone() if patch_embed.bias is not None else None

        out_channels, _, k, _ = weight.shape

        patched_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=patch_embed.stride,  # type: ignore[arg-type]
            padding=patch_embed.padding,  # type: ignore[arg-type]
            bias=(bias is not None),
        )

        with torch.no_grad():
            patched_conv.weight[:, :3, :, :].copy_(weight)
            patched_conv.weight[:, 3:, :, :].zero_()
            if bias is not None and patched_conv.bias is not None:
                patched_conv.bias.copy_(bias)

        patch_embed_container.proj = patched_conv
        logger.info("Patched input layer: 3 -> %d channels (extra initialized to 0)", in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full forward pass and return alpha and foreground predictions.

        Args:
            x: Input tensor with shape [B, 4, H, W] containing RGB + mask channels.

        Returns:
            A dict with two keys:
                "alpha": Predicted alpha matte with shape [B, 1, H, W] in range 0-1.
                "fg": Predicted foreground color with shape [B, 3, H, W] in range 0-1.
        """
        spatial_size = x.shape[2:]

        features = self.encoder(x)

        alpha_logits = self.alpha_decoder(features)
        fg_logits = self.fg_decoder(features)

        alpha_logits_up = functional.interpolate(alpha_logits, size=spatial_size, mode="bilinear", align_corners=False)
        fg_logits_up = functional.interpolate(fg_logits, size=spatial_size, mode="bilinear", align_corners=False)

        alpha_coarse = torch.sigmoid(alpha_logits_up)
        fg_coarse = torch.sigmoid(fg_logits_up)

        rgb_input = x[:, :3, :, :]
        coarse_pred = torch.cat([alpha_coarse, fg_coarse], dim=1)

        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb_input, coarse_pred).to(coarse_pred.dtype)
        else:
            delta_logits = torch.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, 0:1]
        delta_fg = delta_logits[:, 1:4]

        alpha_final = torch.sigmoid(alpha_logits_up + delta_alpha)
        fg_final = torch.sigmoid(fg_logits_up + delta_fg)

        return {"alpha": alpha_final, "fg": fg_final}
