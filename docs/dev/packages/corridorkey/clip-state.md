# Clip State Machine

Every clip in CorridorKey moves through a strict state machine. The state determines what the pipeline will do with a clip and which operations are valid at any given moment.

## The Six States

`EXTRACTING` - A source video has been registered but frames have not been extracted yet. The clip is waiting for FFmpeg to unpack it into the `Frames/` directory.

`RAW` - Frames exist in `Frames/` but no alpha hint has been generated. The clip needs an alpha generator before inference can run.

`MASKED` - A VideoMaMa mask hint exists in `VideoMamaMaskHint/`. The clip is waiting for VideoMaMa to convert the mask into an alpha hint.

`READY` - An alpha hint exists in `AlphaHint/` covering all input frames. The clip is ready for inference.

`COMPLETE` - Inference has run and all output frames are written. The clip is done.

`ERROR` - A processing step failed. The `error_message` field on `ClipEntry` describes what went wrong.

## Valid Transitions

```text
EXTRACTING -> RAW        extraction completes
EXTRACTING -> ERROR      extraction fails

RAW        -> MASKED     user provides a VideoMaMa mask
RAW        -> READY      alpha generator produces AlphaHint
RAW        -> ERROR      alpha generation or scan fails

MASKED     -> READY      VideoMaMa generates alpha from mask
MASKED     -> ERROR      VideoMaMa fails

READY      -> COMPLETE   inference succeeds
READY      -> ERROR      inference fails

ERROR      -> RAW        retry from scratch
ERROR      -> MASKED     retry with mask
ERROR      -> READY      retry inference only
ERROR      -> EXTRACTING retry extraction

COMPLETE   -> READY      reprocess with different params
```

Any transition not listed above raises `InvalidStateTransitionError`.

## How the Pipeline Routes Clips

`process_directory` and `_process_clip` use the state to decide what to do:

| State | Has alpha generator | Action |
|---|---|---|
| `COMPLETE` | any | Skip |
| `ERROR` | any | Skip (log warning) |
| `EXTRACTING` | any | Skip (not ready yet) |
| `RAW` or `MASKED` | no | Skip (log warning: no alpha generator) |
| `RAW` or `MASKED` | yes | Run alpha generator, then inference |
| `READY` | any | Run inference directly |

## State Recovery on Restart

When `find_assets()` scans a clip directory, it sets the state based on what is present on disk. This means the pipeline always recovers to the furthest completed stage after a crash or restart - no work is lost.

Recovery priority (highest first):

1. `COMPLETE` - all input frames have matching output frames (checked via the run manifest)
2. `READY` - `AlphaHint/` covers all input frames
3. `MASKED` - `VideoMamaMaskHint/` exists
4. `EXTRACTING` - a video source exists but no frame sequence yet
5. `RAW` - frame sequence exists, no alpha or mask

## Why a State Machine

The state machine exists to make the pipeline restartable and idempotent. Because state is derived entirely from what is on disk, the pipeline never needs a database or external process registry. A crash at any point leaves the clip in a well-defined state that the next run can resume from.

The strict transition table also makes it impossible for the pipeline to skip a required stage. A clip cannot reach `READY` without an alpha hint, and cannot reach `COMPLETE` without passing through `READY`.

## Related

- [clip-state reference](../../api/corridorkey/clip-state.md)
- [errors reference](../../api/corridorkey/errors.md)
- [Project layout](project-layout.md)
