"""Unit tests for the AlphaGenerator protocol.

AlphaGenerator is a structural Protocol - any object with the right
attributes satisfies it without inheriting from a base class. Tests verify
that a correct implementation is recognised, and that objects missing
either required attribute are correctly rejected by isinstance().
"""

from __future__ import annotations

from corridorkey.protocols import AlphaGenerator


class _ConcreteGenerator:
    """Minimal valid implementation of AlphaGenerator."""

    @property
    def name(self) -> str:
        return "test_gen"

    def generate(self, clip, on_progress=None, on_warning=None) -> None:
        pass


class _MissingName:
    def generate(self, clip, on_progress=None, on_warning=None) -> None:
        pass


class _MissingGenerate:
    @property
    def name(self) -> str:
        return "x"


class TestAlphaGeneratorProtocol:
    """AlphaGenerator protocol - structural subtype checking via isinstance()."""

    def test_valid_implementation_is_instance(self):
        """An object with both name and generate must satisfy the protocol."""
        gen = _ConcreteGenerator()
        assert isinstance(gen, AlphaGenerator)

    def test_missing_name_not_instance(self):
        """An object without a name property must not satisfy the protocol."""
        assert not isinstance(_MissingName(), AlphaGenerator)

    def test_missing_generate_not_instance(self):
        """An object without a generate method must not satisfy the protocol."""
        assert not isinstance(_MissingGenerate(), AlphaGenerator)

    def test_name_returns_string(self):
        """name must return a non-empty string identifying the generator."""
        gen = _ConcreteGenerator()
        assert isinstance(gen.name, str)
        assert gen.name == "test_gen"

    def test_generate_callable(self):
        """generate() must be callable and must not raise on a minimal invocation."""
        gen = _ConcreteGenerator()
        # Should not raise
        gen.generate(None)
