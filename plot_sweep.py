"""
Turns a sweep_rescue.py CSV into the two figures the floor is chosen on.

    python plot_sweep.py --csv sweep_vibe_v1.csv

Figure 1 (decision) answers "which floor, and does the rescue earn its place":
    A  Dice against the floor, one line per rescue margin
    B  what the rescue is worth, PAIRED per case -- the between-case variance
       cancels, which the two means in A cannot show
    C  reference/prediction centerline ratio: are the two sides cutting the
       same tree? 1.0 is the target, not the maximum
    D  how much of the reference is left to score

Figure 2 (dispersion) is the same Dice one case at a time, because a mean over
34 subjects hides a cohort that splits in two.

Read A with C. A rising Dice is a smaller and easier region, not a better
segmentation -- the floor changes WHICH vessels are scored. The floor to pick
is where C sits near 1.0 and A has gone flat, so the number would not have
moved had you chosen 0.5 mm either side.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# validated categorical slots 1 and 2 (light surface #fcfcfb): adjacent CVD
# dE 24.7, normal-vision 33.6, both >= 3:1 on the surface
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def ui_font():
    """The first system sans actually installed; matplotlib warns per glyph otherwise."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Helvetica Neue", "Helvetica", "Segoe UI", "Inter", "Arial", "DejaVu Sans"):
        if name in have:
            return name
    return "sans-serif"


def style():
    plt.rcParams.update({
        "font.family": ui_font(),
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.titlepad": 10,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.fontsize": 9,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "lines.linewidth": 2.0, "lines.markersize": 4.5,
    })


def frame(ax, ylabel=None):
    """Recessive chrome: horizontal hairlines only, two spines."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def label_margin(margin):
    return "sans rattrapage" if margin == 0 else f"rattrapage {margin:g} mm"


def by_floor(data, margins, column, how="mean"):
    """One (floors, values) pair per margin, aggregated over cases."""
    out = []
    for margin in margins:
        grouped = data[data.margin_mm == margin].groupby("min_diameter_mm")[column]
        series = grouped.mean() if how == "mean" else grouped.median()
        out.append((series.index.to_numpy(float), series.to_numpy(float)))
    return out


def panel_dice(ax, data, margins):
    tracks = by_floor(data, margins, "dice_large")
    ends = [y[-1] for _, y in tracks]
    for i, ((x, y), margin) in enumerate(zip(tracks, margins)):
        ax.plot(x, y, "-o", color=SERIES[i], label=label_margin(margin), zorder=3)
        # direct-labelled inside the axes: identity never rests on colour alone,
        # and the four panels keep one x-scale
        above = y[-1] >= max(ends)
        ax.annotate(label_margin(margin), (x[-1], y[-1]),
                    xytext=(-8, 9 if above else -16), textcoords="offset points",
                    color=SERIES[i], fontsize=8.5, ha="right", fontweight="bold")
    frame(ax, "Dice (moyenne sur les cas)")
    ax.set_title("A  Dice des gros vaisseaux")


def panel_paired(ax, data, margins):
    """The rescue's worth, case by case: the between-case variance cancels."""
    base, top = margins[0], margins[-1]
    key = ["case", "class", "min_diameter_mm"]
    wide = data.pivot_table(index=key, columns="margin_mm", values="dice_large")
    if base not in wide or top not in wide:
        ax.set_visible(False)
        return
    delta = (wide[top] - wide[base]).rename("delta").reset_index()

    for _, group in delta.groupby(["case", "class"]):
        group = group.sort_values("min_diameter_mm")
        ax.plot(group.min_diameter_mm, group.delta, color=SERIES[1],
                alpha=0.22, linewidth=1.0, zorder=2)
    middle = delta.groupby("min_diameter_mm").delta.median()
    ax.plot(middle.index, middle.to_numpy(), "-o", color=SERIES[1], zorder=4,
            label="médiane")
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    ax.annotate("aucun effet", (0.01, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=8)
    frame(ax, "Δ Dice par cas")
    ax.set_title(f"B  Ce que rapporte le rattrapage ({top:g} mm vs {base:g})")


def panel_ratio(ax, data, margins):
    for i, margin in enumerate(margins):
        subset = data[data.margin_mm == margin].groupby("min_diameter_mm")
        ratio = subset.reference_kept_mm.mean() / subset.prediction_kept_mm.mean()
        ax.plot(ratio.index, ratio.to_numpy(), "-o", color=SERIES[i],
                label=label_margin(margin), zorder=3)
    ax.axhline(1.0, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("même arbre des deux côtés", (0.01, 1.0),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", color=MUTED, fontsize=8)
    frame(ax, "centerline référence / prédiction")
    ax.set_title("C  Les deux côtés coupent-ils la même anatomie ?")


def panel_kept(ax, data, margins):
    for i, ((x, y), margin) in enumerate(
            zip(by_floor(data, margins, "kept_fraction_reference"), margins)):
        ax.plot(x, 100.0 * y, "-o", color=SERIES[i], label=label_margin(margin), zorder=3)
    frame(ax, "volume de référence conservé (%)")
    ax.set_title("D  Ce qu'il reste à noter")


def figure_decision(data, margins, title, highlight):
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), sharex=True)
    panel_dice(axes[0][0], data, margins)
    panel_paired(axes[0][1], data, margins)
    panel_ratio(axes[1][0], data, margins)
    panel_kept(axes[1][1], data, margins)

    for ax in axes.ravel():
        if not ax.get_visible():
            continue
        if highlight is not None:
            ax.axvline(highlight, color=MUTED, linewidth=1.0, linestyle=(0, (2, 3)), zorder=0)
        ax.set_xlabel("plancher de calibre (mm)")
        ax.tick_params(labelbottom=True)  # each panel is read on its own
    if highlight is not None:
        axes[0][0].annotate(f"{highlight:g} mm", (highlight, 0.02),
                            xycoords=("data", "axes fraction"), xytext=(5, 0),
                            textcoords="offset points", color=MUTED, fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels),
                  bbox_to_anchor=(0.5, 0.005))
    figure.suptitle(title, x=0.008, y=0.982, ha="left", va="top",
                    fontsize=13, fontweight="bold", color=INK)
    figure.text(0.008, 0.947,
                "Un Dice qui monte est une région plus petite et plus facile, pas une "
                "meilleure segmentation : lire A avec C et D.",
                ha="left", va="top", fontsize=9, color=INK_2)
    figure.tight_layout(rect=(0, 0.045, 1, 0.928))
    return figure


def figure_dispersion(data, margins, title, worst):
    figure, axes = plt.subplots(1, len(margins), figsize=(5.6 * len(margins), 4.6),
                               sharey=True, squeeze=False)
    for i, (ax, margin) in enumerate(zip(axes[0], margins)):
        subset = data[data.margin_mm == margin]
        for _, group in subset.groupby(["case", "class"]):
            group = group.sort_values("min_diameter_mm")
            ax.plot(group.min_diameter_mm, group.dice_large, color=SERIES[i],
                    alpha=0.18, linewidth=1.0, zorder=2)

        # name the cases that drag the mean down, at the finest floor
        finest = subset[subset.min_diameter_mm == subset.min_diameter_mm.min()]
        for rank, (_, row) in enumerate(finest.nsmallest(worst, "dice_large").iterrows()):
            track = subset[subset.case == row.case].sort_values("min_diameter_mm")
            ax.plot(track.min_diameter_mm, track.dice_large, color=SERIES[i],
                    alpha=0.9, linewidth=1.4, zorder=3)
            # staggered, because the weakest cases sit close together
            ax.annotate(row.case.replace(".nii.gz", ""), (row.min_diameter_mm, row.dice_large),
                        xytext=(4, -3 - 9 * (rank % 2)), textcoords="offset points",
                        fontsize=7.5, color=INK_2)

        median = subset.groupby("min_diameter_mm").dice_large.median()
        ax.plot(median.index, median.to_numpy(), "-o", color=INK, zorder=5, label="médiane")
        frame(ax, "Dice des gros vaisseaux" if i == 0 else None)
        ax.set_xlabel("plancher de calibre (mm)")
        ax.set_title(f"{label_margin(margin)}  ·  n = {subset.case.nunique()} cas")
        ax.legend(loc="lower right")

    figure.suptitle(title, x=0.008, y=0.985, ha="left", va="top",
                    fontsize=13, fontweight="bold", color=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="output of sweep_rescue.py --csv")
    parser.add_argument("--out", default="sweep", metavar="PREFIX",
                        help="writes PREFIX_decision.png and PREFIX_dispersion.png")
    parser.add_argument("--step", type=float, default=None, metavar="MM",
                        help="cut step to keep when the sweep varied it. Default: the "
                             "only one present, or the most frequent")
    parser.add_argument("--support", default=None, help="mask | centerline. Default as --step")
    parser.add_argument("--peel", default=None, metavar="REF/PRED",
                        help="terminal peel to keep when the sweep varied it (sweep_rescue.py "
                             "--peels), written as it is printed: 1/0, 1/1, 0/0. Default as "
                             "--step. The curves are a floor-against-Dice reading and averaging "
                             "two peels into one would compare trees of different extents")
    parser.add_argument("--class-name", default=None, metavar="NAME",
                        help="keep one class when the sweep scored several")
    parser.add_argument("--highlight", type=float, default=None, metavar="MM",
                        help="draw a rule at the floor you are proposing")
    parser.add_argument("--worst", type=int, default=3, metavar="N",
                        help="name the N weakest cases in the dispersion figure")
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def pick(data, column, requested):
    """One value on an axis the sweep may or may not have varied."""
    present = sorted(data[column].dropna().unique())
    if requested is not None:
        if requested not in present:
            raise SystemExit(f"--{column} {requested} absent, CSV has {present}")
        return requested
    if len(present) > 1:
        chosen = data[column].mode().iloc[0]
        print(f"{column}: {present} in the CSV, keeping {chosen} "
              f"(pass it explicitly to change)")
        return chosen
    return present[0]


def main():
    args = build_parser().parse_args()
    style()
    data = pd.read_csv(args.csv)

    data = data[data.cut_step_mm == pick(data, "cut_step_mm", args.step)]
    data = data[data.support == pick(data, "support", args.support)]
    # written by sweep_rescue.py since --peels put the peel in the grid; older
    # CSVs have neither column and have exactly one peel by construction
    if {"peel_terminals_reference", "peel_terminals_prediction"} <= set(data.columns):
        data = data.assign(peel=data.peel_terminals_reference.astype(str) + "/"
                           + data.peel_terminals_prediction.astype(str))
        data = data[data.peel == pick(data, "peel", args.peel)]
    if args.class_name is not None:
        data = data[data["class"] == args.class_name]

    margins = sorted(data.margin_mm.unique())
    if len(margins) > 2:
        margins = [margins[0], margins[-1]]
        print(f"margins: keeping {margins} -- two lines read, five do not")
    cases, model = data.case.nunique(), (data.model.dropna().iloc[0]
                                         if data.model.notna().any() else "modèle")
    title = f"{model} — {cases} cas, découpe {data.cut_step_mm.iloc[0]:g} mm"

    decision = f"{args.out}_decision.png"
    figure_decision(data, margins, title, args.highlight).savefig(decision, dpi=args.dpi)
    dispersion = f"{args.out}_dispersion.png"
    figure_dispersion(data, margins, title, args.worst).savefig(dispersion, dpi=args.dpi)
    print(f"wrote {decision} and {dispersion}")

    print(f"\n{'floor':>7}{'margin':>8}{'dice':>9}{'ratio':>8}{'kept':>8}{'n':>5}")
    for margin in margins:
        for floor, group in data[data.margin_mm == margin].groupby("min_diameter_mm"):
            ratio = group.reference_kept_mm.mean() / group.prediction_kept_mm.mean()
            print(f"{floor:>7.1f}{margin:>8.2f}{group.dice_large.mean():>9.4f}"
                  f"{ratio:>8.2f}{group.kept_fraction_reference.mean():>7.1%}"
                  f"{group.case.nunique():>5}")


if __name__ == "__main__":
    main()
