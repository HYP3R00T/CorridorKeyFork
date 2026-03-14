# Project Layout

CorridorKey organises footage into projects on disk. Understanding the layout is useful when integrating with external tools, writing custom scanners, or debugging missing assets.

## v2 Layout (Current)

```text
Projects/
    260301_093000_Woman_Jumps/          project folder (timestamped)
        project.json                    project-level metadata
        clips/
            Woman_Jumps/                clip folder (ClipEntry.root_path)
                Source/
                    Woman_Jumps.mp4     original source video
                Frames/                 extracted input frames (PNG/EXR)
                AlphaHint/              alpha hint frames written by a generator
                VideoMamaMaskHint/      optional VideoMaMa mask hint
                Output/
                    FG/                 foreground frames
                    Matte/              alpha matte frames
                    Comp/               preview composite frames
                    Processed/          linear premultiplied RGBA frames
                    .corridorkey_manifest.json
                clip.json               per-clip metadata
            Man_Walks/
                ...
```

The project folder name is `{YYMMDD_HHMMSS}_{sanitized_stem}`. The timestamp ensures uniqueness when the same source file is imported twice.

## v1 Layout (Legacy)

v1 projects have no `clips/` subdirectory. The project folder itself is the clip root:

```text
Projects/
    Woman_Jumps/
        Input/                          input frames (or Input.mp4)
        AlphaHint/
        Output/FG/ Matte/ Comp/
        project.json
```

v1 projects are still fully supported. `is_v2_project()` returns `False` for them and `get_clip_dirs()` returns `[project_dir]` as a single-clip fallback.

## JSON Metadata Files

### project.json

Stored at the project root. Contains project-level metadata.

```json
{
  "version": 2,
  "created": "2026-03-15T09:30:00",
  "display_name": "Woman Jumps",
  "clips": ["Woman_Jumps", "Man_Walks"]
}
```

### clip.json

Stored at the clip root. Contains per-clip metadata including the original source path and in/out range.

```json
{
  "source": {
    "original_path": "/footage/raw/Woman_Jumps.mp4",
    "filename": "Woman_Jumps.mp4",
    "copied": true
  },
  "display_name": "Woman Jumps",
  "in_out_range": {
    "in_point": 24,
    "out_point": 287
  }
}
```

### .corridorkey_manifest.json

Written by `run_inference` into `Output/`. Records which outputs were enabled and the inference parameters used. Used by `completed_stems()` to determine which frames are fully done.

```json
{
  "version": 1,
  "enabled_outputs": ["fg", "matte", "comp", "processed"],
  "formats": {
    "fg": "exr",
    "matte": "exr",
    "comp": "png",
    "processed": "exr"
  },
  "params": {
    "despill_strength": 1.0,
    "auto_despeckle": true,
    "despeckle_size": 400,
    "refiner_scale": 1.0,
    "input_is_linear": false
  }
}
```

## Source Copying vs Referencing

When a clip is added to a project, the source video can either be copied into `Source/` or left in place with only a path reference stored in `clip.json`. Copying is the default and makes the project self-contained. Referencing is useful when footage lives on a shared network drive and copying would be wasteful, but it means the project breaks if the source path changes.

## In/Out Ranges

An in/out range restricts inference to a sub-range of frames. It is stored in `clip.json` and loaded automatically when the clip is scanned. This is useful for long clips where only a portion needs to be processed, or when the source video has handles that should be excluded.

The range is inclusive on both ends. A clip with `in_point=24, out_point=287` processes 264 frames.

## Related

- [project reference](../../api/corridorkey/project.md)
- [clip-state reference](../../api/corridorkey/clip-state.md)
- [Clip state machine](clip-state.md)
