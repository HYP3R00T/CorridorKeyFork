# Configuration

CorridorKey is configured through a single `corridorkey.toml` file. Each pipeline stage has its own section. Settings that are not specified use sensible defaults.

Run `ck config --write` to write the current configuration with all defaults to disk.

## Device

The `device` setting controls which hardware runs the neural network. The default is `auto`, which selects the best available device at startup: AMD GPU (ROCm) if present, then NVIDIA GPU (CUDA), then Apple Silicon (MPS), then CPU.

GPU inference is significantly faster than CPU. On a modern GPU, a single frame at 2048 resolution takes roughly 100-300ms. On CPU, the same frame takes several seconds.

## Model Resolution

The model runs at a fixed square resolution. The native training resolution is 2048x2048, which produces the best output quality. Smaller resolutions (1536, 1024) reduce VRAM usage at the cost of some quality, particularly in fine edge detail.

Setting `img_size = 0` (the default) selects the resolution automatically based on available VRAM: below 6 GB selects 1024, 6-12 GB selects 1536, 12 GB or more selects 2048.

## The Refiner

The CNN refiner corrects edge detail after the main transformer forward pass. It is enabled by default. Disabling it is faster but produces visibly coarser alpha matte edges.

The refiner can run in two modes. In `full_frame` mode it processes the entire image at once, which is fastest on GPUs with enough VRAM. In `tiled` mode it processes the image in overlapping 512x512 tiles, which keeps peak VRAM usage flat and is required on low-VRAM GPUs. The output quality is identical for both modes.

Setting `refiner_mode = "auto"` (the default) selects the mode based on available VRAM: below 12 GB uses tiled, 12 GB or more uses full frame.

## Floating Point Precision

The model can run in float32, float16, or bfloat16. Lower precision uses less VRAM and runs faster, at the cost of numerical accuracy.

Setting `model_precision = "auto"` (the default) selects the best format for the hardware: bfloat16 on Ampere+ GPUs and Apple Silicon, float16 on older GPUs, float32 on CPU. See [The Neural Network](../pipeline/neural-network.md) for why bfloat16 is preferred over float16 on modern hardware.

## Green Spill Removal

The `despill_strength` setting controls how aggressively green spill is removed from the foreground. A value of 0.0 disables despill entirely. A value of 1.0 applies full suppression. The default is 0.5.

For subjects with naturally green elements, reducing this value avoids over-correcting. See [Green Spill](../pipeline/green-spill.md) for the underlying concept.

## Source Passthrough

The model predicts foreground colour for every pixel. In the solid interior of the subject, the original source pixels are a better estimate than the model's prediction. Source passthrough replaces the model's foreground in opaque interior regions with the original source pixels.

Both `preprocess.source_passthrough` and `postprocess.source_passthrough` must be enabled for this to work. Both default to `true`.

## Output Formats

Each output type (alpha, foreground, processed RGBA, composite preview) can be independently enabled or disabled and written as PNG or EXR.

PNG is 8-bit and suitable for previews. EXR is 32-bit floating point and is the correct format for production compositing outputs. See [Colour Space](../pipeline/colour-space.md) for why full precision matters for compositing.

The default format for all outputs is PNG. Switch to EXR for production work where full precision matters.

## Full Parameter Reference

The complete list of parameters with their defaults and valid values is in the [Developer Configuration reference](../../dev/packages/corridorkey/configuration.md).
