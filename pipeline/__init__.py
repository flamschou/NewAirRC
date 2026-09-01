"""Training and inference.

    config -> manifest -> transforms -> dataset -> train -> model

`config` is the single source of truth for paths, patch geometry, class
names and hyperparameters; every other module reads it rather than
hardcoding values.
"""
