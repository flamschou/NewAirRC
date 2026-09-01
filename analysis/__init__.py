"""Analysis of a segmentation, once one exists.

    centerline    skeleton, tree, branching ratios
    truncate      cutting a tree back to its large vessels
    compute_dice / sweep_rescue    scoring, whole tree and large vessels
    compare_predictions            two checkpoints against each other
    connectivity  how fragmented a mask is, in one number
    phantom / calibrate / radius_audit    synthetic trees of known geometry,
                                          which the chain is calibrated and
                                          audited against
    cohort        per-subject ratio files assembled into one table
"""
