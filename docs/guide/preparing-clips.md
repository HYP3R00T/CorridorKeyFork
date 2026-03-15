# Preparing Clips

CorridorKey processes clips that are in the `READY` state. A clip reaches `READY` when it has both input frames and a matching set of alpha hint frames on disk. You provide both.

## Required Folder Structure

Each clip must be its own folder containing two subfolders:

```text
my_shot/
    Frames/
        frame_0001.png
        frame_0002.png
        frame_0003.png
        ...
    AlphaHint/
        frame_0001.png
        frame_0002.png
        frame_0003.png
        ...
```

`Frames/` holds the green screen input frames. `AlphaHint/` holds the corresponding alpha matte frames, one per input frame. The frame count in both folders must match exactly. If `AlphaHint/` has fewer frames than `Frames/`, the clip stays in `RAW` state and will not be processed.

## Multiple Clips

To process several shots in one session, put each clip in its own subfolder and point the wizard at the parent:

```text
session/
    actor_jump/
        Frames/
            frame_0001.png
            ...
        AlphaHint/
            frame_0001.png
            ...
    product_spin/
        Frames/
            frame_0001.png
            ...
        AlphaHint/
            frame_0001.png
            ...
```

```shell
corridorkey wizard /path/to/session
```

The wizard will find both clips and show them in the state table.

## Alpha Hint Format

The alpha hint frames are greyscale images where white (255) is fully opaque and black (0) is fully transparent. PNG is the recommended format. EXR is also accepted.

The alpha hint does not need to be a perfect matte. It is used as a hint to guide the model. A rough garbage matte or a simple threshold mask is sufficient.

## Supported Input Formats

Input frames in `Frames/` can be PNG, JPEG, TIFF, or OpenEXR. All frames in a clip must use the same format and resolution.

For best quality use PNG or EXR. JPEG compression introduces artefacts that can affect matte edges.

If your input frames are linear light (e.g. EXR from a camera or renderer), enable `input_is_linear` in the wizard settings.

## Frame Naming

Frames must be numbered sequentially. Any consistent zero-padded naming works:

- `frame_0001.png`, `frame_0002.png` ...
- `shot_001.exr`, `shot_002.exr` ...
- `0001.png`, `0002.png` ...

The names in `Frames/` and `AlphaHint/` do not need to match each other. The pipeline matches frames by sort order, not by name.

## Checking Clip State

Run `corridorkey scan` to verify your clips are in `READY` state before processing:

```shell
corridorkey scan /path/to/session
```

A clip showing `RAW` means `AlphaHint/` is missing or empty. A clip showing `READY` is good to go.

## Related

- [Processing clips](processing.md)
- [Outputs](outputs.md)
- [Clip state machine](../dev/packages/corridorkey/clip-state.md)
