"""Dev entry point — runs the pipeline directly without the CLI layer.

Useful for quick local testing without installing ckcli.
For normal use, run: ck process <clips_dir>
"""

from __future__ import annotations

from pathlib import Path

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")


def main() -> None:
    from utilityhub_config import ensure_config_file

    from corridorkey_new import detect_gpu, load, load_config, resolve_device, scan, setup_logging
    from corridorkey_new.infra import APP_NAME
    from corridorkey_new.infra.config import CorridorKeyConfig
    from corridorkey_new.infra.model_hub import ensure_model
    from corridorkey_new.runtime import PipelineConfig, PipelineRunner
    from corridorkey_new.stages.inference import load_model

    ensure_config_file(CorridorKeyConfig(), APP_NAME)
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    print(gpu.model_dump_json(indent=2))

    clips = scan(CLIPS_DIR)
    manifest = load(clips[0])
    print(manifest.model_dump_json(indent=2))

    device = resolve_device(config.device)
    inference_config = config.to_inference_config(device=device)
    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    model = load_model(inference_config)

    pipeline_config = PipelineConfig(
        preprocess=config.to_preprocess_config(device=device, resolved_img_size=inference_config.img_size),
        inference=inference_config,
        model=model,
    )

    PipelineRunner(manifest, pipeline_config).run()
    print(f"Done. Output: {manifest.output_dir}")


if __name__ == "__main__":
    main()
