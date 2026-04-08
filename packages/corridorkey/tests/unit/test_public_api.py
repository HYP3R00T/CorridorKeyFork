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


class TestClipEntryExported:
    def test_clip_entry_in_all(self):
        assert "ClipEntry" in corridorkey.__all__

    def test_clip_entry_is_class(self):
        from corridorkey import ClipEntry

        assert isinstance(ClipEntry, type)

    def test_clip_entry_constructible_from_clip(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import Clip, ClipEntry, ClipState

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame_000001.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipEntry.from_clip(clip)

        assert entry.name == "test"
        assert entry.state == ClipState.RAW
        assert entry.manifest is None
        assert entry.in_out_range is None

    def test_clip_entry_state_machine_accessible(self, tmp_path):
        from corridorkey import Clip, ClipEntry, ClipState

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipEntry(clip=clip, state=ClipState.RAW)
        entry.transition_to(ClipState.READY)
        assert entry.state == ClipState.READY


class TestInOutRangeExported:
    def test_in_out_range_in_all(self):
        assert "InOutRange" in corridorkey.__all__

    def test_in_out_range_is_class(self):
        from corridorkey import InOutRange

        assert isinstance(InOutRange, type)

    def test_in_out_range_constructible(self):
        from corridorkey import InOutRange

        r = InOutRange(in_point=5, out_point=14)
        assert r.frame_count == 10
        assert r.contains(5)
        assert r.contains(14)
        assert not r.contains(15)

    def test_in_out_range_to_frame_range(self):
        from corridorkey import InOutRange

        r = InOutRange(in_point=0, out_point=9)
        assert r.to_frame_range() == (0, 10)

    def test_in_out_range_usable_on_clip_entry(self, tmp_path):
        from corridorkey import Clip, ClipEntry, ClipState, InOutRange

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipEntry(clip=clip, state=ClipState.RAW)
        entry.in_out_range = InOutRange(in_point=10, out_point=49)
        assert entry.in_out_range.frame_count == 40


class TestResolveClipStateExported:
    def test_resolve_clip_state_in_all(self):
        assert "resolve_clip_state" in corridorkey.__all__

    def test_resolve_clip_state_callable(self):
        from corridorkey import resolve_clip_state

        assert callable(resolve_clip_state)

    def test_resolve_clip_state_returns_clip_state(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import Clip, ClipState, resolve_clip_state

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame_000001.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        state = resolve_clip_state(clip)

        assert isinstance(state, ClipState)
        assert state == ClipState.RAW
