#!/usr/bin/env python3
"""Strip plot des metriques morphometriques pour presentation.

Un panneau par metrique, un point par sujet, bande de reference en fond.
Les sujets dont le fit repose sur 3 points sont traces en creux et exclus
de la moyenne (mais restent visibles).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- donnees

SUBJECTS = ["FLAWAN", "ADOTHO", "ARIINE", "BROLAU", "CAUTIM", "CHAALI",
            "GHAITH", "HERQUE", "HOTHI", "ISHOUN", "JOULAU", "KERNIN",
            "LAFMAR", "LECELI", "LERMAX", "MENMAR", "LOOBER", "MULTHI",
            "NDIELI", "OZIMAR", "PASALE", "REIANN", "SAMARM", "SAVMAR"]

R_D = [1.510, 1.441, 1.500, 1.440, 1.465, 1.441, 1.490, 1.476,
       1.451, 1.529, 1.810, 1.495, 1.616, 1.347, 1.487, 1.563,
       1.459, 1.482, 1.431, 1.579, 1.564, 1.421, 1.585, 1.412]

R_B = [2.713, 2.867, 2.976, 2.398, 3.400, 2.382, 2.570, 2.672,
       2.865, 2.606, 4.074, 2.912, 2.820, 2.511, 2.822, 3.347,
       2.837, 2.533, 2.292, 3.493, 2.790, 2.589, 3.189, 3.464]

MURRAY = [2.89, 3.44, 2.45, 3.90, 2.87, 2.99, 2.91, 2.89,
          3.72, 2.98, 2.76, 3.18, 2.83, 3.10, 4.24, 2.44,
          2.39, 3.16, 2.76, 3.13, 3.74, 3.94, 3.00, 3.30]

# fits a 3 points : exclus des moyennes des ratios, gardes pour Murray
EXCLUDED = {"JOULAU", "MENMAR", "OZIMAR", "SAMARM", "SAVMAR"}

# ---------------------------------------------------------------- style

INK = "#1a1a1a"
DOT = "#2b6cb0"
DOT_WEAK = "#a0aec0"
BAND = "#e8b4b8"
MEAN = "#c53030"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
})


def strip(ax, values, mask_excl, ref_lo, ref_hi, title, subtitle,
          use_all=False, arrow_up=False):
    """Un panneau. mask_excl : True = fit a 3 points."""
    rng = np.random.default_rng(4)
    x = rng.uniform(-0.16, 0.16, len(values))

    # bande de reference
    if ref_lo != ref_hi:
        ax.axhspan(ref_lo, ref_hi, color=BAND, alpha=0.45, zorder=0, lw=0)
    else:
        ax.axhline(ref_lo, color=BAND, lw=3, zorder=0, alpha=0.9)

    kept = [v for v, e in zip(values, mask_excl) if use_all or not e]

    for xi, v, excl in zip(x, values, mask_excl):
        if excl and not use_all:
            ax.plot(xi, v, "o", ms=6, mfc="none", mec=DOT_WEAK,
                    mew=1.2, zorder=2)
        else:
            ax.plot(xi, v, "o", ms=6, color=DOT, alpha=0.75, zorder=3)

    m, s = np.mean(kept), np.std(kept, ddof=1)
    ax.errorbar(0.42, m, yerr=s, fmt="_", ms=22, mew=2.5,
                color=MEAN, ecolor=MEAN, elinewidth=1.8,
                capsize=6, capthick=1.8, zorder=4)
    ax.text(0.60, m, f"{m:.2f}\n±{s:.2f}", va="center", ha="left",
            fontsize=10, color=MEAN, linespacing=1.35)

    if arrow_up:
        ax.annotate("", xy=(0.42, m + s + 0.32), xytext=(0.42, m + s + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=MEAN,
                                    lw=1.6, mutation_scale=14))
        ax.text(0.30, m + s + 0.36, "saturation", fontsize=9,
                color=MEAN, ha="center", style="italic")

    ax.set_xlim(-0.45, 1.35)
    ax.set_xticks([])
    ax.set_title(title, fontsize=13, pad=14, weight="bold")
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9.5, color="#666666")
    ax.grid(axis="y", color="#e2e8f0", lw=0.7, zorder=-1)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
excl = [s in EXCLUDED for s in SUBJECTS]

strip(axes[0], R_D, excl, 1.56, 1.60,
      "$R_d$  rapport de diamètre",
      "réf. 1,56–1,60   ·   n = 19")

strip(axes[1], R_B, excl, 3.03, 3.03,
      "$R_b$  rapport de ramification",
      "réf. 3,03   ·   n = 19   ·   borne basse",
      arrow_up=True)

strip(axes[2], MURRAY, excl, 3.0, 3.0,
      "exposant de Murray",
      "optimum 3   ·   n = 24",
      use_all=True)

axes[0].set_ylabel("valeur mesurée")
fig.text(0.5, -0.02,
         "Points creux : fit à 3 points, exclus de la moyenne.   "
         "$R_l$ non estimable (R² < 0,7 chez 12/24, non récupérable sur fantôme).",
         ha="center", fontsize=9, color="#666666")

fig.tight_layout()
fig.savefig("ratios_cohorte.png",
            dpi=220, bbox_inches="tight", facecolor="white")

# ------------------------------------------------------- verification
for name, vals, allsub in [("R_d", R_D, False), ("R_b", R_B, False),
                           ("Murray", MURRAY, True)]:
    kept = [v for v, e in zip(vals, excl) if allsub or not e]
    print(f"{name:8s} n={len(kept):2d}  moyenne={np.mean(kept):.3f}  "
          f"ET={np.std(kept, ddof=1):.3f}  "
          f"étendue={min(kept):.3f}–{max(kept):.3f}")