"""Smoke tests for the public API surface.

Verifies that every symbol in corridorkey.__all__ is importable and that
key additions from recent interface work are present and functional.
"""

from __future__ import annotations

import corridorkey


class TestAllSymbolsResolve:
    def test_all_symbols_importable(self):
        missing = [name for name in corridorkey.__all__ if getattr(corridorkey, name, None) is None]
        assert missing == [], f"Symbols in __all__ that do not resolve: {missing}"


class TestVersionExposed:
    def test_version_in_all(self):
        assert "__version__" in corridorkey.__all__

    def test_version_is_string(self):
        assert isinstance(corridorkey.__version__, str)

    def test_version_not_empty(self):
        assert corridorkey.__version__ != ""


class TestRunnerExported:
    def test_runner_in_all(self):
        assert "Runner" in corridorkey.__all__

    def test_runner_is_class(self):
        from corridorkey import Runner

        assert isinstance(Runner, type)


class TestGetFrameFilesExported:
    def test_get_frame_files_in_all(self):
        assert "get_frame_files" in corridorkey.__all__

    def test_get_frame_files_callable(self):
        from corridorkey import get_frame_files

        assert callable(get_frame_files)

    def test_get_frame_files_returns_list(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import get_frame_files

        # Empty directory returns empty list
        assert get_frame_files(tmp_path) == []

        # Directory with images returns sorted list
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"frame_{i:06d}.png"), np.zeros((4, 4, 3), dtype=np.uint8))

        files = get_frame_files(tmp_path)
        assert len(files) == 3
        assert [f.name for f in files] == [
            "frame_000000.png",
            "frame_000001.png",
            "frame_000002.png",
        ]
