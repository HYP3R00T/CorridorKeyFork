"""CorridorKey - application layer for the CorridorKey AI chroma keying pipeline."""

from corridorkey.clip_state import (
    ClipAsset,
    ClipEntry,
    ClipState,
    scan_clips_dir,
    scan_project_clips,
)
from corridorkey.config import CorridorKeyConfig, export_config, load_config
from corridorkey.contracts import FrameResult, InferenceParams, OutputConfig, WriteConfig
from corridorkey.errors import CorridorKeyError, FFmpegNotFoundError
from corridorkey.ffmpeg_tools import check_ffmpeg
from corridorkey.frame_io import FrameData, load_frame
from corridorkey.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType
from corridorkey.model_manager import MODEL_DOWNLOAD_URL, MODEL_FILENAME, download_model, is_model_present
from corridorkey.models import InOutRange
from corridorkey.pipeline import ClipSummary, PipelineResult, process_directory
from corridorkey.project import (
    add_clips_to_project,
    create_project,
    detect_unstructured,
    get_clip_dirs,
    get_display_name,
    is_image_file,
    is_v2_project,
    is_video_file,
    load_in_out_range,
    organize_clips,
    read_clip_json,
    read_project_json,
    save_in_out_range,
    set_display_name,
    write_clip_json,
    write_project_json,
)
from corridorkey.protocols import AlphaGenerator
from corridorkey.service import (
    CorridorKeyService,
    inference_params_to_postprocess,
    output_config_to_write_config,
)
from corridorkey.validators import ValidationResult, validate_job_inputs
from corridorkey.writer import generate_masks, write_outputs

__all__ = [
    # Validation
    "validate_job_inputs",
    "ValidationResult",
    # Service - primary entry point for CLI, GUI, and server consumers
    "CorridorKeyService",
    "InferenceParams",
    "OutputConfig",
    "FrameResult",
    # Bridge helpers - convert application types to core stage contracts
    "inference_params_to_postprocess",
    "output_config_to_write_config",
    # Pipeline stages (I/O and orchestration)
    "load_frame",
    "generate_masks",
    "write_outputs",
    # Stage contracts
    "FrameData",
    "WriteConfig",
    # Config
    "CorridorKeyConfig",
    "load_config",
    "export_config",
    # Protocols
    "AlphaGenerator",
    # Pipeline - high-level batch processing
    "process_directory",
    "PipelineResult",
    "ClipSummary",
    # Clip state
    "ClipAsset",
    "ClipEntry",
    "ClipState",
    "scan_clips_dir",
    "scan_project_clips",
    # Models
    "InOutRange",
    # Job queue - async GPU job management (GUI use)
    "GPUJob",
    "GPUJobQueue",
    "JobType",
    "JobStatus",
    # Errors
    "CorridorKeyError",
    "FFmpegNotFoundError",
    # FFmpeg diagnostics
    "check_ffmpeg",
    # Model management
    "is_model_present",
    "download_model",
    "MODEL_DOWNLOAD_URL",
    "MODEL_FILENAME",
    # Project management
    "create_project",
    "add_clips_to_project",
    "detect_unstructured",
    "organize_clips",
    "get_clip_dirs",
    "is_v2_project",
    "write_project_json",
    "read_project_json",
    "write_clip_json",
    "read_clip_json",
    "get_display_name",
    "set_display_name",
    "save_in_out_range",
    "load_in_out_range",
    "is_video_file",
    "is_image_file",
]
