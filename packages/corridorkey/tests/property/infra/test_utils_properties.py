"""Property-based tests for corridorkey.infra.utils."""

from __future__ import annotations

from corridorkey.infra.utils import natural_sort_key
from hypothesis import given
from hypothesis import strategies as st

# Strings that look like typical frame filenames
_frame_name = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-."), min_size=1, max_size=40
)


class TestNaturalSortKeyProperties:
    @given(st.lists(_frame_name, min_size=1, max_size=20))
    def test_sort_is_stable_on_resort(self, names: list[str]):
        """Sorting twice produces the same result as sorting once."""
        once = sorted(names, key=natural_sort_key)
        twice = sorted(once, key=natural_sort_key)
        assert once == twice

    @given(st.lists(_frame_name, min_size=2, max_size=20))
    def test_sort_is_total(self, names: list[str]):
        """Every pair of names can be compared without error."""
        result = sorted(names, key=natural_sort_key)
        assert len(result) == len(names)

    @given(st.integers(min_value=1, max_value=999), st.integers(min_value=1, max_value=999))
    def test_numeric_ordering(self, a: int, b: int):
        """frame_N sorts before frame_M iff N < M."""
        name_a = f"frame_{a}.png"
        name_b = f"frame_{b}.png"
        result = sorted([name_a, name_b], key=natural_sort_key)
        if a < b:
            assert result == [name_a, name_b]
        elif a > b:
            assert result == [name_b, name_a]
        else:
            assert result[0] == result[1]

    @given(_frame_name)
    def test_key_is_list(self, name: str):
        """natural_sort_key always returns a list."""
        assert isinstance(natural_sort_key(name), list)
