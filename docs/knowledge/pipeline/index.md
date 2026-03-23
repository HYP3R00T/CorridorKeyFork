# Pipeline

CorridorKey processes footage through six sequential stages. Each stage transforms data from one form into another and passes it to the next.

## The Six Stages

| Stage | Name | What it produces |
|---|---|---|
| 0 | Scan | A list of clips discovered from a path. |
| 1 | Load | A manifest describing a clip's frames, alpha hints, and output location. |
| 2 | Preprocess | A normalised tensor ready for the neural network. |
| 3 | Infer | Raw alpha and foreground predictions from the model. |
| 4 | Postprocess | Refined, compositing-ready output arrays at source resolution. |
| 5 | Write Outputs | Image files on disk. |

## Two Kinds of Stages

Stages 0, 1, and 5 interact with the filesystem. They discover clips, extract frames from video, and write output images.

Stages 2, 3, and 4 are pure computation. They receive data in memory and return data in memory. No files are read or written. This separation means the compute stages can be understood and tested independently of any I/O concerns.

## Assembly Line

The three compute stages run concurrently. While the GPU is running inference on one frame, the CPU is reading the next frame and writing the previous one. This keeps the GPU busy continuously rather than waiting for I/O between frames.

## Concepts Behind the Stages

Each stage is built on a set of underlying ideas. Understanding those ideas helps you interpret the outputs and tune the settings.

- [Alpha Matting](alpha-matting.md) - What the model is actually solving and why it is hard.
- [Colour Space](colour-space.md) - Why sRGB, linear light, and premultiplied alpha each exist and when each is correct.
- [Green Spill](green-spill.md) - Why green light contaminates the subject and how it is removed.
- [The Neural Network](neural-network.md) - How GreenFormer works: encoder, decoders, and the CNN refiner.
