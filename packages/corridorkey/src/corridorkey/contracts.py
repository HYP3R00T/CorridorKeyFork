"""Data contracts for the corridorkey processing layer.

InferenceParams  - parameters for a single inference job.
OutputConfig     - which output types to produce and their format.
FrameResult      - per-frame result summary (no numpy arrays).
WriteConfig      - parameters controlling write_outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class InferenceParams:
    """Frozen parameters for a single inference job.

    Attributes:
        input_is_linear: True if the input frames are in linear light (e.g. EXR).
        despill_strength: Strength of the green-spill suppression (0.0-1.0).
        auto_despeckle: Enable automatic matte despeckling.
        despeckle_size: Maximum speckle area in pixels to remove.
        refiner_scale: Scale factor passed to the optional refiner stage.
        source_passthrough: Pass original source pixels through in opaque interior
            regions. Only the edge transition band uses the model's fg prediction.
        edge_erode_px: Pixels to erode the interior mask inward before blending.
        edge_blur_px: Gaussian blur radius for the transition blend seam.
    """

    input_is_linear: bool = False
    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    refiner_scale: float = 1.0
    source_passthrough: bool = False
    edge_erode_px: int = 3
    edge_blur_px: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> InferenceParams:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OutputConfig:
    """Which output types to produce and their format.

    Attributes:
        fg_enabled: Write foreground (RGBA) frames.
        fg_format: File format for FG frames ("exr" or "png").
        matte_enabled: Write alpha matte frames.
        matte_format: File format for matte frames ("exr" or "png").
        comp_enabled: Write composited preview frames.
        comp_format: File format for comp frames ("exr" or "png").
        processed_enabled: Write pre-processed input frames.
        processed_format: File format for processed frames ("exr" or "png").
        exr_compression: EXR compression codec for all EXR outputs.
            "dwaa" - lossy DCT, visually lossless, ~5x faster writes (default).
            "pxr24" - lossless 24-bit, larger files, slower.
            "zip" - lossless ZIP deflate, widely compatible.
            "none" - uncompressed, maximum compatibility, largest files.
    """

    fg_enabled: bool = True
    fg_format: str = "exr"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"
    exr_compression: str = "dwaa"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OutputConfig:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def enabled_outputs(self) -> list[str]:
        out = []
        if self.fg_enabled:
            out.append("fg")
        if self.matte_enabled:
            out.append("matte")
        if self.comp_enabled:
            out.append("comp")
        if self.processed_enabled:
            out.append("processed")
        return out


@dataclass
class FrameResult:
    """Result summary for a single processed frame (no numpy arrays).

    Attributes:
        frame_index: Zero-based index of the frame within the clip.
        input_stem: Filename stem of the input frame (e.g. "frame_000001").
        success: True if the frame was processed and written successfully.
        warning: Non-fatal message if the frame was skipped or had issues.
    """

    frame_index: int
    input_stem: str
    success: bool
    warning: str | None = None


@dataclass
class WriteConfig:
    """Parameters controlling write_outputs.

    Mirrors OutputConfig without coupling to it, so this can be constructed
    independently by the GUI or tests.

    Attributes:
        fg_enabled: Write foreground RGB frames.
        fg_format: "exr" or "png".
        matte_enabled: Write alpha matte frames.
        matte_format: "exr" or "png".
        comp_enabled: Write composite preview frames.
        comp_format: "exr" or "png".
        processed_enabled: Write linear premultiplied RGBA frames.
        processed_format: "exr" (only valid option for compositing delivery).
        exr_compression: EXR compression codec ("dwaa", "piz", "zip", "none").
        dirs: Absolute directory paths keyed by output name.
            Keys: "fg", "matte", "comp", "processed".
    """

    fg_enabled: bool = True
    fg_format: str = "exr"
    matte_enabled: bool = True
    matte_format: str = "exr"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "exr"
    exr_compression: str = "dwaa"
    dirs: dict[str, str] = field(default_factory=dict)
