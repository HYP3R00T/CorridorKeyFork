from pathlib import Path

from corridorkey_new import detect_gpu, load, load_config, resolve_alpha, scan, setup_logging

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs")


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
    print(f"Found {len(clips)} clip(s)")

    manifest = load(clips[0])
    print(manifest.model_dump_json(indent=2))

    if manifest.needs_alpha:
        # Alpha generation is the interface's responsibility.
        # Generate externally, then hand the result back to the pipeline.
        alpha_dir = _generate_alpha_externally(manifest)
        manifest = resolve_alpha(manifest, alpha_dir)
        print(f"alpha resolved: {manifest.alpha_frames_dir}")

    print(manifest.model_dump_json(indent=2))
