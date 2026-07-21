# -*- coding: utf-8 -*-
"""
check_class_balance.py

Counts, across the whole manifest, how many volumes actually contain
each raw label value (config.LABEL_CLASS_MAP keys) and how many voxels
of each. Used to check whether "artery" (or any class) is so rare that
it's effectively invisible during training/validation.
"""
import nibabel as nib
import numpy as np

import config as cfg
import manifest as manifest_module

entries = manifest_module.load_manifest(cfg.MANIFEST_PATH)

totals = {raw: 0 for raw in cfg.LABEL_CLASS_MAP}
volumes_with_class = {raw: 0 for raw in cfg.LABEL_CLASS_MAP}

for entry in entries:
    label = np.asarray(nib.load(entry["label"]).dataobj)
    for raw_value, class_index in cfg.LABEL_CLASS_MAP.items():
        count = int((label == raw_value).sum())
        totals[raw_value] += count
        if count > 0:
            volumes_with_class[raw_value] += 1

print(f"{len(entries)} volumes in manifest")
for raw_value, class_index in cfg.LABEL_CLASS_MAP.items():
    name = cfg.CLASS_NAMES[class_index]
    print(
        f"raw={raw_value} ({name}): {totals[raw_value]} voxels total, "
        f"present in {volumes_with_class[raw_value]}/{len(entries)} volumes"
    )
