# Naming Conventions

This document defines naming conventions for all Python identifiers in this repository.
All conventions follow standard Python style (PEP 8) with project-specific rules layered on top.

## Packages

Use lowercase with hyphens. Names must be prefixed with `corridorkey-` to namespace them
within the project, followed by a noun or noun phrase describing the layer or domain.

(GOOD)

- `corridorkey-core` - the core inference layer
- `corridorkey-app` - the application orchestration layer
- `corridorkey-cli` - the command-line interface

(AVOID)

- `core` - not namespaced, will conflict with other packages
- `corridorkey_core` - use hyphens for package names, underscores are for Python module names
- `CorridorKeyCore` - wrong case

Note: the Python import name (the `src/` folder) uses underscores as required by Python
(e.g. `corridorkey_core`), but the installable package name in `pyproject.toml` uses hyphens
(e.g. `corridorkey-core`).

## Modules (Files)

Use lowercase with underscores. Names must be gerunds or nouns that describe what the module does, not what it contains.

(GOOD)

- `compositing.py` - performs compositing operations
- `inference_engine.py` - runs model inference
- `model_transformer.py` - defines the transformer model

(AVOID)

- `utils.py` - too generic, says nothing about purpose
- `helpers.py` - same problem
- `ColorUtils.py` - wrong case

## Classes

Use `PascalCase`. Names must be nouns that describe what the class represents.

(GOOD)

- `CorridorKeyEngine`
- `GreenFormer`
- `DecoderHead`

(AVOID)

- `corridorKeyEngine` - wrong case
- `ProcessFrames` - verb, not a noun
- `Mgr` - abbreviations that obscure meaning

## Functions and Methods

Use `snake_case`. Public functions must be verbs or verb phrases that describe what the function does.
Private functions (prefixed with `_`) follow the same rule.

(GOOD)

- `process_frame`
- `linear_to_srgb`
- `clean_matte`
- `_load_model`

(AVOID)

- `processFrame` - wrong case
- `frame` - not a verb
- `do_stuff` - not descriptive

## Constants

Use `UPPER_SNAKE_CASE`. Module-private constants are prefixed with `_`.
Each constant must have a comment on the line above it explaining what it represents.

(GOOD)

```python
# Slope of the linear segment
_SRGB_LINEAR_SCALE = 12.92
```

(AVOID)

```python
_SRGB_LINEAR_SCALE = 12.92  # inline comment
LINEAR_SCALE = 12.92        # missing private prefix for module-internal constants
scale = 12.92               # wrong case, not descriptive
```

## Variables

Use `snake_case`. Names must be descriptive nouns. Single-letter names are only acceptable as loop indices or in mathematical expressions that mirror a well-known formula.

(GOOD)

- `processed_alpha`
- `fg_despilled_lin`
- `checkpoint_path`

(AVOID)

- `processedAlpha` - wrong case
- `pa` - abbreviation that obscures meaning
- `temp` - not descriptive

## Parameters

Use `snake_case`. Follow the same rules as variables. Boolean parameters must be phrased as `is_`, `use_`, `has_`, or `auto_` prefixes.

(GOOD)

- `input_is_linear`
- `use_refiner`
- `auto_despeckle`

(AVOID)

- `linear` - ambiguous, not clearly boolean
- `refiner` - reads as a noun, not a flag

## Type Aliases

Use `PascalCase` when defining a named type alias.

(GOOD)

```python
ImageArray = np.ndarray
```

(AVOID)

```python
image_array = np.ndarray  # looks like a variable
```

## Related

- [Documentation Principles](documentation-principles/index.md)
- [Authoring Documentation](authoring-documentation.md)
