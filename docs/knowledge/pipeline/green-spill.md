# Green Spill

Green spill is the contamination of the foreground subject by green light reflected from the backdrop. It appears as a green tint on the subject's edges, hair, and any reflective surfaces facing the screen.

## Why It Happens

A green screen backdrop reflects green light in all directions. Some of that light bounces off the subject. The camera records this reflected green light as part of the subject's colour, even though it is not part of the subject's actual appearance.

The effect is strongest at edges, where the subject is physically close to the backdrop and the reflected light is most intense. It is also visible on reflective surfaces (glasses, jewellery, wet skin) and on light-coloured clothing that picks up the ambient green cast.

## Why It Matters for Matting

Spill affects both the alpha matte and the foreground colour.

In the foreground colour, spill appears as a green fringe around the subject. When the subject is composited over a non-green background, this fringe is visible and looks unnatural.

In the alpha matte, spill can cause the model to underestimate the alpha value at edges, because the green-tinted pixels look more like background than foreground. This produces semi-transparent edges where the subject should be fully opaque.

## How Spill Removal Works

Spill removal compares the green channel to the average of the red and blue channels. Any excess green beyond what red and blue would predict is considered spill. That excess is redistributed equally to red and blue to neutralise the tint without darkening the subject:

\[
g_{\text{limit}} = \frac{r + b}{2}, \quad \Delta g = \max(g - g_{\text{limit}},\ 0)
\]

\[
g' = g - \Delta g, \quad r' = r + \frac{\Delta g}{2}, \quad b' = b + \frac{\Delta g}{2}
\]

This preserves the overall luminance of the pixel while removing the green cast. The strength of the correction is configurable: 0 applies no correction, 1 applies the full calculated correction.

## Limits of Spill Removal

Spill removal is a heuristic. It works well for moderate spill on neutral-coloured subjects. It can produce incorrect results when:

- The subject has naturally green elements (plants, green clothing). The algorithm cannot distinguish natural green from spill.
- The spill is very heavy and extends deep into the subject interior. Removing it fully may shift the subject's colours noticeably.
- The backdrop is not evenly lit and the spill colour varies across the frame.

For difficult cases, reducing the despill strength and accepting some residual spill is often better than over-correcting and shifting the subject's colours.

## Related Documents

- [Colour Space](colour-space.md) - Spill removal operates in sRGB. Understanding the colour space helps interpret the results.
- [Alpha Matting](alpha-matting.md) - How spill affects the alpha estimate at edges.
