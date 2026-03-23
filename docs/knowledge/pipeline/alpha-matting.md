# Alpha Matting

Alpha matting is the problem of separating a foreground subject from its background at the pixel level, including semi-transparent regions like hair, motion blur, and fine fabric.

## What an Alpha Matte Is

Every pixel in an image can be described as a mixture of foreground and background:

\[
C = \alpha \cdot F + (1 - \alpha) \cdot B
\]

where \(C\) is the observed colour, \(F\) is the foreground colour, \(B\) is the background colour, and \(\alpha\) is the coverage of the foreground at that pixel, ranging from 0 (fully background) to 1 (fully foreground).

The alpha matte is the image of \(\alpha\) values across all pixels. White means fully foreground. Black means fully background. Grey means partial coverage, which occurs at edges, in hair strands, and wherever the subject is semi-transparent.

## Why It Is Hard

For a fully opaque pixel, the observed colour is just the foreground colour. The alpha is 1.0 and the background does not contribute.

For a semi-transparent pixel, the observed colour is a blend of foreground and background. Given only the observed colour \(C\), recovering both \(\alpha\) and \(F\) is an underdetermined problem: there are infinitely many combinations of \(\alpha\) and \(F\) that produce the same \(C\). Additional information is needed to constrain the solution.

A green screen provides that constraint. Because the background colour \(B\) is known (it is green), the equation has fewer unknowns. The model can use the known background colour to estimate \(\alpha\) and \(F\) more reliably.

## The Alpha Hint

Even with a known background colour, the problem is still difficult at edges. The alpha hint is a rough mask that tells the model where the foreground is likely to be. It does not need to be precise. A coarse separation between subject and background is enough.

The model uses the hint as a spatial prior. It focuses its attention on the boundary region between foreground and background, where the matting problem is hardest, and uses the hint to anchor its estimate of which side is which.

## Keying vs Matting

Chroma keying and alpha matting solve the same problem but in different ways.

A chroma key selects pixels based on their colour. Pixels close to the key colour are made transparent. This works well for clean, evenly lit green screens but fails at edges, in semi-transparent regions, and wherever the subject reflects the backdrop.

Alpha matting uses a neural network trained on many examples of foreground subjects against various backgrounds. It learns to estimate \(\alpha\) and \(F\) jointly, including in the difficult semi-transparent regions where chroma keying breaks down.

## Related Documents

- [The Neural Network](neural-network.md) - How GreenFormer estimates alpha and foreground colour.
- [Colour Space](colour-space.md) - How alpha values interact with colour space in the outputs.
