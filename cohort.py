# -*- coding: utf-8 -*-
"""
cohort.py

Assembles the per-subject ratio files of a cohort into one table, and runs
the two checks that decide whether the subjects in it are comparable.

The fit range is not the same for every subject and cannot be. The diameter
floor is mechanical -- three voxels of the coarse axis -- but it lands on a
different order at every spacing, so a finer acquisition keeps more orders
than a coarser one. That is the shape of the data, not a defect of the
method, and the only thing it requires is that it be declared: the RULE is
pre-specified, the RANGE is not, and a reader given the ratios alone cannot
tell a difference in anatomy from a difference in resolution.

Hence the table, one row per subject with its spacing, its anisotropy, the
floor that was applied and the orders that survived it, next to the ratios.
And hence the two checks:

    the anisotropy should be near-constant across a cohort acquired on one
    protocol. A subject that departs from it was acquired differently, or has
    a header that says so wrongly, and either way its ratios are not
    comparable to the rest -- its floor is somewhere else, so its fit rests
    on different orders of the same tree.

    no subject may fall under three orders. Three points still admit a
    regression, if barely; two do not admit one at all, and a slope through
    two points is an arithmetic identity rather than a fit. A subject under
    the bar is dropped, with the reason recorded, not fitted anyway.

It reads what `centerline.py --ratios-csv` writes and does no analysis of its
own beyond those checks -- deliberately, so that nothing here can disagree
with what produced the numbers.

Usage:
    python cohort.py 'results/*_ratios.csv' --counting element --out cohort.csv
    python cohort.py results/*.csv --ratio R_d --min-orders 3
"""
import argparse
import csv
import glob
import os

import numpy as np

RATIOS = ("R_b", "R_d", "R_l")


def load(paths, counting, ordering=None):
    """
    Every ratio row of every file, keyed by subject.

    A file whose ordering differs from the rest is a hard error rather than a
    warning: generation-counted ratios are mechanically smaller than Strahler
    ones on the very same tree, so mixing the two produces a cohort spread
    that is entirely an artefact of how the runs were invoked.

    The count floor is refused the same way and for the same reason. It is
    the second rule that decides which orders enter the fit -- the diameter
    floor censors the thin end, --fit-min-branches the trunk end -- and two
    subjects run under two values of it are fitted over ranges that are not
    the same measurement, whatever their spacings agree on.
    """
    subjects, orderings, floors = {}, set(), set()
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("counting") != counting:
                    continue
                if ordering and row.get("ordering") != ordering:
                    continue
                orderings.add(row.get("ordering"))
                if row.get("fit_min_branches") not in (None, ""):
                    floors.add(row["fit_min_branches"])
                name = row.get("subject") or os.path.basename(path)
                subjects.setdefault(name, {"subject": name, "path": path, "ratios": {}})
                subjects[name]["ratios"][row["ratio"]] = row
                for column in ("spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "anisotropy",
                               "fit_floor_mm", "fit_min_branches", "order_min", "order_max",
                               "n_orders", "prespecified"):
                    if row.get(column) not in (None, ""):
                        subjects[name][column] = row[column]
    if len(orderings) > 1:
        raise SystemExit(f"the files mix orderings {sorted(orderings)}; ratios counted in "
                         f"generations and in Strahler orders are not comparable on the same "
                         f"tree. Re-run the odd ones, or select one with --ordering")
    if len(floors) > 1:
        raise SystemExit(f"the files mix --fit-min-branches {sorted(floors)}; that floor moves "
                         f"which orders enter the fit, so ratios computed under two values of it "
                         f"do not rest on comparable ranges. Re-run the odd ones")
    return list(subjects.values()), (orderings.pop() if orderings else "")


def number(row, key):
    """A stored column as a float, or None."""
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def outliers(values, tolerance):
    """
    Indices whose value departs from the cohort median by more than
    `tolerance`, relatively.

    Median and not mean: the check exists to find the subject acquired on a
    different protocol, and that subject would drag a mean towards itself and
    hide inside the spread it created.
    """
    finite = [v for v in values if v is not None]
    if not finite:
        return [], None
    middle = float(np.median(finite))
    if middle <= 0:
        return [], middle
    return [i for i, v in enumerate(values)
            if v is not None and abs(v / middle - 1.0) > tolerance], middle


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+",
                        help="The --ratios-csv files, or globs matching them")
    parser.add_argument("--counting", choices=("segment", "element"), default="element",
                        help="Which counting to read. Default: element")
    parser.add_argument("--ordering", default=None,
                        help="Keep only rows with this ordering, when a directory holds several")
    parser.add_argument("--min-orders", type=int, default=3, metavar="N",
                        help="A subject whose fit rests on fewer orders than this is listed as "
                             "excluded. Three points still admit a regression; two make the slope "
                             "an identity rather than a fit. Default: 3")
    parser.add_argument("--anisotropy-tolerance", type=float, default=0.05, metavar="FRACTION",
                        help="How far a subject's anisotropy may sit from the cohort median before "
                             "it is flagged as acquired differently. Default: 0.05")
    parser.add_argument("--out", help="CSV to write, one row per subject")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.paths for p in (glob.glob(pattern) or [pattern])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"no such file: {missing[0]}")
    subjects, ordering = load(paths, args.counting)
    if not subjects:
        raise SystemExit(f"no rows with counting={args.counting} in {len(paths)} file(s)")
    subjects.sort(key=lambda s: s["subject"])
    print(f"{len(subjects)} subject(s) from {len(paths)} file(s), {ordering} ordering, "
          f"{args.counting}s")

    ratio_values = np.array([[number(s["ratios"].get(name, {}), "value") or np.nan
                              for name in RATIOS] for s in subjects], dtype=float)
    anisotropy = [number(s, "anisotropy") for s in subjects]
    orders = [number(s, "n_orders") for s in subjects]

    print("\n=== per subject ===")
    print("  subject                    spacing (mm)        aniso   floor    orders   "
          + "  ".join(f"{name:>7}" for name in RATIOS))
    for index, subject in enumerate(subjects):
        grid = [number(subject, f"spacing_{axis}_mm") for axis in "xyz"]
        grid_text = ("        ?        " if None in grid
                     else "/".join(f"{v:.3f}" for v in grid))
        floor = number(subject, "fit_floor_mm")
        span = (f"{subject.get('order_min', '?')}..{subject.get('order_max', '?')}"
                f" ({int(orders[index]) if orders[index] else 0})")
        print(f"  {subject['subject'][:24]:<24} {grid_text:>19} "
              f"{anisotropy[index] or float('nan'):7.2f} "
              f"{floor or float('nan'):7.2f} {span:>9}   "
              + "  ".join(f"{v:7.3f}" if np.isfinite(v) else "      -"
                          for v in ratio_values[index]))

    print("\n=== check 1: is the cohort one protocol ===")
    flagged, middle = outliers(anisotropy, args.anisotropy_tolerance)
    if middle is None:
        print("  no anisotropy recorded -- these files predate the columns; re-run centerline.py")
    else:
        print(f"  median anisotropy {middle:.2f}:1, "
              f"range {min(v for v in anisotropy if v is not None):.2f} to "
              f"{max(v for v in anisotropy if v is not None):.2f}")
        if not flagged:
            print(f"  every subject sits within {args.anisotropy_tolerance:.0%} of it: one "
                  f"protocol, and the floor lands on comparable orders throughout")
        for index in flagged:
            print(f"  FLAG {subjects[index]['subject']}: {anisotropy[index]:.2f}:1 against a median "
                  f"of {middle:.2f}:1. A different acquisition, or a header that misreports one. "
                  f"Its floor is elsewhere, so its fit rests on different orders of the same tree "
                  f"and its ratios are not comparable to the rest")

    print("\n=== check 2: does every subject support a fit ===")
    short = [i for i, n in enumerate(orders) if n is None or n < args.min_orders]
    if not short:
        print(f"  every subject rests on at least {args.min_orders} orders "
              f"({int(min(n for n in orders if n))} at worst)")
    for index in short:
        count = int(orders[index]) if orders[index] else 0
        print(f"  EXCLUDE {subjects[index]['subject']}: {count} order(s) clear a floor of "
              f"{number(subjects[index], 'fit_floor_mm') or float('nan'):.2f} mm. Drop it and say "
              f"so -- a slope through {count} point(s) is not a measurement")

    spread = [n for n in orders if n is not None]
    if spread and min(spread) != max(spread):
        print(f"\n  the fit rests on {int(min(spread))} to {int(max(spread))} orders across the "
              f"cohort. That is the resolution differing between subjects, not the anatomy, and it "
              f"has to be declared with the ratios: the RULE fixing the range was pre-specified, "
              f"the range itself was not and moves from subject to subject")
    if any(number(s, "prespecified") != 1.0 for s in subjects):
        print(f"\n  WARNING: {sum(1 for s in subjects if number(s, 'prespecified') != 1.0)} "
              f"subject(s) did not declare --prespecified. Their fit range is recorded as chosen "
              f"after the per-order table was visible, which is the one degree of freedom that "
              f"turns any tree into a published value")

    if args.out:
        columns = ("subject", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "anisotropy",
                   "fit_floor_mm", "fit_min_branches", "order_min", "order_max", "n_orders",
                   "prespecified", "excluded", "flagged") + tuple(
                       f"{name}{suffix}" for name in RATIOS
                       for suffix in ("", "_ci_low", "_ci_high", "_r2"))
        rows = []
        for index, subject in enumerate(subjects):
            row = {column: subject.get(column, "") for column in columns if column in subject}
            row["subject"] = subject["subject"]
            row["excluded"] = int(index in short)
            row["flagged"] = int(index in flagged)
            for name in RATIOS:
                source = subject["ratios"].get(name, {})
                row[name] = source.get("value", "")
                row[f"{name}_ci_low"] = source.get("ci_low", "")
                row[f"{name}_ci_high"] = source.get("ci_high", "")
                row[f"{name}_r2"] = source.get("r2", "")
            rows.append({column: row.get(column, "") for column in columns})
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
