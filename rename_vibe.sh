#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/flamant/data/ct/lidc_idri"
DRY_RUN=1

if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=0
fi

for dir in "$ROOT"/LIDC-IDRI-*/; do
    id=$(basename "$dir")
    src="${dir}generated_vibe.nii.gz"
    dst="${dir}generated_vibe_MRI_${id}.nii.gz"

    if [[ -f "$src" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[DRY RUN] mv \"$src\" \"$dst\""
        else
            mv -n "$src" "$dst"
            echo "renamed: $src -> $dst"
        fi
    fi
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "Dry run only, aucun fichier renomme. Relance avec --apply pour effectuer le renommage."
fi
