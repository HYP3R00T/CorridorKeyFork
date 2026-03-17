"""Replace common non-ASCII characters with ASCII equivalents across all source files."""

import pathlib

REPLACEMENTS = [
    ("\u2014", "--"),  # em dash
    ("\u2013", "-"),  # en dash
    ("\u00d7", "x"),  # multiplication sign x
    ("\u2192", "->"),  # right arrow
    ("\u2190", "<-"),  # left arrow
    ("\u2265", ">="),  # greater or equal
    ("\u2264", "<="),  # less or equal
    ("\u2248", "~="),  # approximately equal
    ("\u00b7", "*"),  # middle dot
    ("\u2019", "'"),  # right single quote
    ("\u2018", "'"),  # left single quote
    ("\u201c", '"'),  # left double quote
    ("\u201d", '"'),  # right double quote
    ("\u00b0", " deg"),  # degree sign
    ("\u2026", "..."),  # ellipsis
    ("\u00e9", "e"),  # e acute
    ("\u00e0", "a"),  # a grave
]

EXTENSIONS = ("*.md", "*.py", "*.toml")
EXCLUDE = (".venv", "site", ".git", "__pycache__", "Temp")


def main() -> None:
    fixed = []
    for ext in EXTENSIONS:
        for path in pathlib.Path(".").rglob(ext):
            if any(part in str(path) for part in EXCLUDE):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            new_text = text
            for char, replacement in REPLACEMENTS:
                new_text = new_text.replace(char, replacement)
            # Replace any remaining non-ASCII with '?'
            new_text = new_text.encode("ascii", errors="replace").decode("ascii")
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                fixed.append(str(path))

    for f in fixed:
        print(f"Fixed: {f}")
    print(f"Done. {len(fixed)} file(s) updated.")


if __name__ == "__main__":
    main()
