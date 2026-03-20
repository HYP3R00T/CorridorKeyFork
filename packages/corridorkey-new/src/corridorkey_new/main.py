from pathlib import Path

from corridorkey_new import (
    detect_gpu,
    load,
    load_config,
    resolve_alpha,
    resolve_device,
    scan,
    setup_logging,
)
from corridorkey_new.pipeline import PipelineConfig, PipelineRunner
from corridorkey_new.preprocessor import PreprocessConfig

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")


def _generate_alpha_externally(manifest) -> Path:
    """Stub: simulate external alpha generation.

    In a real CLI/GUI this would invoke the alpha generator tool, wait for it
    to finish, and return the path to the generated alpha frames directory.

    For now, prompt the user to provide the path manually.
    """
    print(f"  Alpha required for '{manifest.clip_name}'.")
    print(f"  Run your alpha generator on: {manifest.frames_dir}")
    raw = input("  Enter path to generated alpha frames directory: ").strip()
    return Path(raw)


def main() -> None:
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    print(gpu.model_dump_json(indent=2))

    clips = scan(CLIPS_DIR)

    manifest = load(clips[0])
    print(manifest.model_dump_json(indent=2))

    if manifest.needs_alpha:
        alpha_dir = _generate_alpha_externally(manifest)
        manifest = resolve_alpha(manifest, alpha_dir)
        print(f"alpha resolved: {manifest.alpha_frames_dir}")

    device = resolve_device(config.device)
    pipeline_config = PipelineConfig(
        preprocess=PreprocessConfig(img_size=2048, device=device),
        input_queue_depth=2,
        output_queue_depth=2,
    )

    print(f"\nRunning pipeline for '{manifest.clip_name}' ({manifest.frame_count} frames)...")
    PipelineRunner(manifest, pipeline_config).run()
    print("Done.")
