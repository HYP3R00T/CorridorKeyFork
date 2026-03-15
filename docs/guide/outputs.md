# Outputs

After inference, CorridorKey writes four sets of output files inside each clip folder. This page explains what each output is and how to use it.

## Output Folder Structure

```text
my_shot/
    frame_0001.png          (original input, untouched)
    ...
    Output/
        FG/
            frame_0001.exr  foreground frames
            ...
        Matte/
            frame_0001.exr  alpha matte frames
            ...
        Comp/
            frame_0001.png  preview composite frames
            ...
        Processed/
            frame_0001.png  processed RGBA frames
            ...
        .corridorkey_manifest.json
```

## The Four Outputs

### FG (Foreground)

The foreground frames contain the subject extracted from the green screen as an RGB image in sRGB colour space, straight (unpremultiplied) alpha. Default format is EXR.

Use FG when you want to composite the subject yourself in a DCC tool using your own background and lighting. Import with straight alpha mode.

### Matte

The matte frames contain the raw alpha prediction as a greyscale image. White is fully opaque, black is fully transparent. Default format is EXR.

Use Matte when you need the alpha channel separately, for example to apply additional matte cleanup in a compositing application before combining with the foreground.

### Comp

The comp frames are a ready-to-view preview composite. The subject is placed over a grey checkerboard in sRGB. Default format is PNG.

Use Comp to quickly review the keying result without opening a compositing application. These frames are not intended for final delivery.

### Processed

The processed frames contain the final RGBA output in linear light, premultiplied. Default format is PNG (configurable to EXR).

Use Processed when you want a single RGBA file ready for import into a compositing application. Import with premultiplied (associated) alpha mode.

## Choosing Output Formats

The default formats (EXR for FG and Matte, PNG for Comp and Processed) suit most workflows. Change them with flags on `corridorkey process`:

```shell
corridorkey process /path/to/clips --fg-format png --matte-format png
```

Or set persistent defaults in `~/.config/corridorkey/corridorkey.yaml`:

```yaml
fg_format: exr
matte_format: exr
comp_format: png
processed_format: exr
```

## Skipping Outputs

If you do not need the composite preview or processed RGBA, skip them to save disk space and speed up processing:

```shell
corridorkey process /path/to/clips --no-comp --no-processed
```

## Importing into Compositing Applications

### After Effects

Import the Processed EXR sequence. In the Import dialog set Footage to "Straight - Unmatted" if using FG, or "Premultiplied - Matted With Color: Black" if using Processed.

### DaVinci Resolve / Fusion

Import the FG EXR sequence with the Matte EXR as a separate alpha channel, or import the Processed EXR directly with premultiplied alpha.

### Nuke

Read the FG and Matte as separate Read nodes, or read the Processed EXR. Set the `premult` knob on the Read node to match the file type.

## The Manifest File

`.corridorkey_manifest.json` inside `Output/` records which outputs were enabled and the inference parameters used. CorridorKey uses this file to determine which frames are complete when resuming an interrupted run. Do not delete it.

## Related

- [Processing clips](processing.md)
- [Configuration](../dev/packages/corridorkey/configuration.md)
- [Project layout](../dev/packages/corridorkey/project-layout.md)
