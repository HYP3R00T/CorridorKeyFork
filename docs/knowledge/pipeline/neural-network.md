# The Neural Network

GreenFormer is the neural network at the core of CorridorKey. It takes a frame and an alpha hint as input and produces an alpha matte and a foreground colour image as output.

## Architecture Overview

GreenFormer has three components: an encoder, two decoders, and a CNN refiner.

The encoder processes the full input and produces a hierarchy of feature representations at different spatial scales. The decoders combine those representations to produce predictions. The refiner corrects edge detail that the encoder and decoders tend to soften.

## The Encoder: Hiera

The encoder is Hiera, a hierarchical vision transformer developed by Meta. It was pretrained on ImageNet and then fine-tuned for matting.

Transformers process images by dividing them into fixed-size patches and computing relationships between all patches simultaneously. This gives the encoder a global view of the image: it can relate a pixel at the top of the frame to a pixel at the bottom, which is important for understanding what is foreground and what is background.

Hiera produces four feature maps at progressively coarser spatial scales. The finest scale captures local detail (edges, textures). The coarsest scale captures global context (the overall shape of the subject, the overall colour of the background).

## The Decoders

Two decoders run in parallel. The alpha decoder produces a coarse alpha matte. The foreground decoder produces a coarse foreground colour image.

Each decoder combines the feature maps from all four encoder scales using a Feature Pyramid Network approach. Coarser features provide context. Finer features provide spatial precision. The decoder upsamples progressively from the coarsest scale to the full model resolution.

The output of each decoder is a set of logits. The sigmoid function converts these to values in the range 0 to 1.

## The CNN Refiner

Transformers process images in patches. At the boundary between patches, fine detail can be lost. This produces blocky artefacts at subject edges in the coarse decoder output.

The CNN refiner corrects this. It is a convolutional neural network that operates at full resolution and specialises in the edge region. It takes the original RGB input alongside the coarse predictions and outputs a delta correction:

\[
\text{output} = \sigma(\text{coarse logits} + \Delta \cdot s)
\]

where \(\sigma\) is the sigmoid function, \(\Delta\) is the refiner's correction, and \(s\) is the refiner scale (1.0 by default). The refiner does not replace the decoder output. It adds a correction to it.

## Why the Four-Channel Input

The model receives four channels: three colour channels plus the alpha hint. The alpha hint is a rough mask indicating where the foreground is likely to be.

Without the hint, the model must infer the foreground location entirely from the image content. With the hint, it has a spatial prior that constrains the problem. The model focuses its attention on the boundary region between foreground and background, where the matting problem is hardest.

## Model Resolution and VRAM

The model runs at a fixed square resolution. The native training resolution is 2048x2048. Running at lower resolutions (1536, 1024) reduces VRAM usage but also reduces the spatial detail available to the encoder, which can soften fine edges.

The refiner is the most VRAM-intensive component because it processes the full-resolution tensor. On GPUs with limited VRAM, the refiner runs in tiled mode: it processes the image in overlapping 512x512 tiles and blends the results back together. The output is identical to the full-frame pass.

## Related Documents

- [Alpha Matting](alpha-matting.md) - The problem the network is solving.
- [Colour Space](colour-space.md) - Why the network requires sRGB input.
