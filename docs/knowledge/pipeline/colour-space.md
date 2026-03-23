# Colour Space

Colour space determines how numerical values in an image correspond to perceived brightness and colour. Using the wrong colour space at any point in the pipeline produces incorrect results that are difficult to diagnose.

## Linear Light vs sRGB

Light behaves linearly in the physical world. Doubling the number of photons doubles the perceived brightness. A value of 0.5 in linear light represents exactly half the brightness of a value of 1.0.

Human vision does not work this way. We are more sensitive to differences in dark tones than in bright tones. sRGB encodes images to match this perceptual sensitivity: more of the available bit depth is allocated to dark values, and less to bright values. A value of 0.5 in sRGB does not represent half the physical brightness of 1.0. It represents roughly 21% of it.

The transfer function that converts between linear light and sRGB is defined by the IEC 61966-2-1 standard. It is a piecewise function: a linear segment near zero and a power curve for the rest.

## Why the Model Requires sRGB

GreenFormer's encoder was pretrained on ImageNet, a dataset of photographs stored in sRGB. The encoder's weights encode patterns in the sRGB distribution. Feeding it linear light images would be like showing it photographs with the wrong gamma applied: the patterns it learned to recognise would not match the input.

EXR files from VFX pipelines are typically in linear light. The preprocessor converts them to sRGB before the model sees them.

## Why Compositing Requires Linear Light

Compositing operations blend colours together. Blending in sRGB produces incorrect results because sRGB is a perceptual encoding, not a physical one. Adding two sRGB values does not produce the correct physical result.

Consider compositing a foreground subject over a background. The correct operation is:

\[
C_{\text{out}} = \alpha \cdot F_{\text{linear}} + (1 - \alpha) \cdot B_{\text{linear}}
\]

This is physically correct only when \(F\) and \(B\) are in linear light. Performing the same operation in sRGB produces colours that are too dark in the blend region.

The premultiplied RGBA output from CorridorKey is in linear light for this reason. NLE and VFX tools expect linear light inputs for compositing.

## Premultiplied Alpha

Premultiplied alpha multiplies the foreground colour by the alpha value at every pixel before storing it:

\[
C_{\text{premult}} = \alpha \cdot F_{\text{linear}}
\]

In fully transparent regions, the stored colour is zero. In semi-transparent regions, the colour is scaled down proportionally to the transparency.

Compositing a premultiplied layer over a background is a single addition:

\[
C_{\text{out}} = C_{\text{premult}} + (1 - \alpha) \cdot B_{\text{linear}}
\]

With straight (unpremultiplied) alpha, an extra multiplication is required first. Premultiplied is the standard format for compositing because the operation is simpler and numerically more stable, particularly in semi-transparent regions where straight alpha can produce values outside the valid range.

## The Four Outputs and Their Colour Spaces

| Output | Colour space | Why |
|---|---|---|
| Alpha matte | Linear | Alpha represents coverage, not colour. No colour space applies. |
| Foreground | sRGB straight | The model predicts in sRGB. Useful for workflows that handle premultiplication separately. |
| Processed RGBA | Linear premultiplied | The correct format for compositing in NLE and VFX tools. |
| Composite preview | sRGB | For display only. Not for compositing. |

## Related Documents

- [Alpha Matting](alpha-matting.md) - What the alpha values represent.
- [Green Spill](green-spill.md) - How spill removal interacts with colour.
