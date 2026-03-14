"""Tests for corridorkey_core.model_transformer.

Instantiates GreenFormer at a small img_size to verify architecture shapes
without requiring a checkpoint or GPU. The sub-module tests (MLP,
DecoderHead, RefinerBlock, CNNRefinerModule) run in the fast suite and
catch regressions in channel counts or tensor shapes that would otherwise
only surface during a full training or inference run.

The full GreenFormer forward pass tests are marked slow because timm model
creation takes a few seconds even at the smallest valid img_size.
"""

import pytest
import torch
from corridorkey_core.model_transformer import (
    MLP,
    CNNRefinerModule,
    DecoderHead,
    GreenFormer,
    RefinerBlock,
)


class TestMLP:
    """MLP projection layer - output shape must match embed_dim."""

    def test_output_shape(self):
        """Output last dimension must equal embed_dim regardless of sequence length."""
        mlp = MLP(input_dim=128, embed_dim=64)
        x = torch.randn(2, 10, 128)
        assert mlp(x).shape == (2, 10, 64)


class TestDecoderHead:
    """Decoder head output channel count for alpha (1) and FG (3) heads."""

    def _make_features(self, batch: int = 1, spatial: int = 8):
        """Synthetic feature maps matching Hiera Base Plus channel counts."""
        return [
            torch.randn(batch, 112, spatial * 4, spatial * 4),
            torch.randn(batch, 224, spatial * 2, spatial * 2),
            torch.randn(batch, 448, spatial, spatial),
            torch.randn(batch, 896, spatial // 2, spatial // 2),
        ]

    def test_alpha_head_output_shape(self):
        """Alpha head must produce a single-channel output."""
        head = DecoderHead(output_dim=1)
        features = self._make_features()
        out = head(features)
        assert out.shape[1] == 1

    def test_fg_head_output_shape(self):
        """FG head must produce a three-channel output."""
        head = DecoderHead(output_dim=3)
        features = self._make_features()
        out = head(features)
        assert out.shape[1] == 3


class TestRefinerBlock:
    """RefinerBlock must preserve spatial dimensions and channel count."""

    def test_output_shape_preserved(self):
        """Output shape must exactly match input shape."""
        block = RefinerBlock(channels=32, dilation=2)
        x = torch.randn(1, 32, 16, 16)
        assert block(x).shape == x.shape


class TestCNNRefinerModule:
    """CNNRefinerModule output shape - takes image + coarse prediction, returns refined."""

    def test_output_shape(self):
        """Output must have the same spatial size as the input and out_channels channels."""
        refiner = CNNRefinerModule(in_channels=7, hidden_channels=32, out_channels=4)
        img = torch.randn(1, 3, 64, 64)
        coarse = torch.randn(1, 4, 64, 64)
        out = refiner(img, coarse)
        assert out.shape == (1, 4, 64, 64)


@pytest.mark.slow
class TestGreenFormer:
    """Full model instantiation and forward pass. Slow due to timm model creation.

    These tests verify the end-to-end tensor contract of GreenFormer without
    a real checkpoint. They catch shape regressions in the encoder/decoder
    wiring that would otherwise only appear during inference.
    """

    IMG_SIZE = 64  # smallest valid size to keep the test fast

    def test_instantiation(self):
        """Model must instantiate without error at the minimum valid img_size."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        assert model is not None

    def test_forward_output_keys(self):
        """Forward pass must return a dict with at least "alpha" and "fg" keys."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert "alpha" in out
        assert "fg" in out

    def test_forward_alpha_shape(self):
        """Alpha output must be [B, 1, H, W]."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].shape == (1, 1, self.IMG_SIZE, self.IMG_SIZE)

    def test_forward_fg_shape(self):
        """FG output must be [B, 3, H, W]."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["fg"].shape == (1, 3, self.IMG_SIZE, self.IMG_SIZE)

    def test_forward_output_range(self):
        """Alpha and FG outputs must be in [0, 1] - they are sigmoid-activated."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=False)
        model.eval()
        x = torch.rand(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].min() >= 0.0
        assert out["alpha"].max() <= 1.0
        assert out["fg"].min() >= 0.0
        assert out["fg"].max() <= 1.0

    def test_forward_with_refiner(self):
        """The optional refiner stage must not change the output shape contract."""
        model = GreenFormer(img_size=self.IMG_SIZE, use_refiner=True)
        model.eval()
        x = torch.zeros(1, 4, self.IMG_SIZE, self.IMG_SIZE)
        with torch.inference_mode():
            out = model(x)
        assert out["alpha"].shape == (1, 1, self.IMG_SIZE, self.IMG_SIZE)
