"""Pipeline stages — stateless transformation functions.

Each subfolder is one stage in the processing pipeline:

    scanner/      stage 0 — discover clips from a path
    loader/       stage 1 — validate, extract, return ClipManifest
    preprocessor/ stage 2 — read frames, GPU transforms, return tensors
    inference/    stage 3 — run model, return alpha + fg tensors
    postprocessor/stage 4 — resize, despill, despeckle, composite
    writer/       stage 5 — write output images to disk

All stages are stateless functions. Import from corridorkey directly,
not from this package.
"""
