from pathlib import Path

from corridorkey_new.entrypoint import scan
from corridorkey_new.infra.config import load_config
from corridorkey_new.infra.device_utils import detect_gpu
from corridorkey_new.infra.logging import setup_logging
from corridorkey_new.loader.manifest import load

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs")


def main() -> None:
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    print(gpu)
    print(type(gpu))

    clips = scan(CLIPS_DIR)
    print(f"Found {len(clips)} clip(s)")
    for clip in clips:
        manifest = load(clip)
        print(manifest)
        print(type(manifest))
