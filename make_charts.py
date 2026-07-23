#!/usr/bin/env python3
"""Render each dashboard chart as a standalone JPEG (matplotlib, Agg).
Run on the box:  uv run --with matplotlib --with pillow python make_charts.py
Data inline (mirrors dashboard.html). Palette matches the site's series colors."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import os

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BG = "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})
OUT = "charts"; os.makedirs(OUT, exist_ok=True)

DEPTHS = [512,1024,2048,4096,8192,16384,32768,49152,65536,98304,131072,196608,262144]
Q6_TG = [54.78,54.39,54.57,53.58,53.93,53.38,52.01,50.87,49.04,47.64,45.58,42.35,38.32]
Q8_TG = [53.68,53.62,53.51,53.35,52.95,52.21,51.0,49.85,48.43,46.84,44.97,41.17,38.99]
Q6_PP = [1392.4,1311.67,1352.97,1366.59,1354.4,1335.02,1294.3,1252.65,1218.29,1141.86,1080.91,969.43,876.32]
Q8_PP = [1457.79,1336.62,1386.54,1387.67,1383.91,1366.48,1324.1,1280.51,1244.92,1168.3,1104.24,988.81,892.6]
SPEC_X = [496,4109,33021,98991,197095,258903]
S_NONE = [51.92,50.73,48.63,44.76,40.15,37.79]
S_MTP  = [92.84,85.83,95.36,78.25,70.36,69.3]
S_NGRAM= [20.92,19.37,18.39,17.01,15.32,17.55]
UB = [256,512,1024,2048,4096]
UB98 = [1080,1144,1189,1194,1162]; UB256 = [843,875,879,840,833]

def ctxfmt(v, _=None):
    v = int(round(v))
    return f"{v//1024}K" if v >= 1024 else str(v)

def save(fig, name):
    fig.tight_layout(pad=1.1)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=160, format="jpg", pil_kwargs={"quality": 92}, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", p)

def logline(x, series, title, ylabel, ymax, name, xt=(512,4096,32768,262144)):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for lbl, ys, c in series:
        ax.plot(x, ys, "-o", color=c, lw=2, ms=4.5, mfc=BG, mec=c, mew=1.6, label=lbl, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlim(min(x)*0.9, max(x)*1.12); ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(FixedLocator(list(xt)))
    ax.xaxis.set_major_formatter(FuncFormatter(ctxfmt))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("context (tokens)"); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best", fontsize=10)
    save(fig, name)

# 1. spec-decode cross
logline(SPEC_X, [("MTP",S_MTP,ORANGE),("none (baseline)",S_NONE,BLUE),("n-gram",S_NGRAM,AQUA)],
        "Speculative decode — TG vs context (Q8, tensor)", "decode tok/s", 105, "01_spec_decode.jpg")
# 2. quant TG
logline(DEPTHS, [("Q8_0",Q8_TG,ORANGE),("Q6_K_XL",Q6_TG,BLUE)],
        "Decode vs context — Q6 vs Q8 (tensor)", "decode tok/s", 62, "02_quant_tg.jpg",
        xt=(512,8192,65536,262144))
# 3. quant PP
logline(DEPTHS, [("Q8_0",Q8_PP,ORANGE),("Q6_K_XL",Q6_PP,BLUE)],
        "Prefill vs context — Q6 vs Q8 (tensor)", "prefill tok/s", 1600, "03_quant_pp.jpg",
        xt=(512,8192,65536,262144))

# 4. comparison bars
fig, ax = plt.subplots(figsize=(6.4, 3.5))
groups = ["Prefill","Decode","Decode+MTP"]; import numpy as np
xi = np.arange(len(groups)); w = 0.38
b1 = ax.bar(xi-w/2, [1392,55,95], w, color=ORANGE, label="4× RTX 3080", zorder=3)
b2 = ax.bar(xi+w/2, [1119,40,40], w, color=BLUE, label="4× RTX 5060 Ti", zorder=3)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+18, f"{int(b.get_height())}",
            ha="center", va="bottom", fontsize=9.5, color="#52514e")
ax.set_xticks(xi); ax.set_xticklabels(groups); ax.set_ylim(0,1650)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("4× RTX 3080 vs 8× RTX 5060 Ti box (tok/s, matched Q6/tensor)", color=INK,
             fontsize=12, fontweight="bold", loc="left", pad=10)
ax.set_ylabel("tok/s"); ax.legend(frameon=False, fontsize=10)
save(fig, "04_comparison.jpg")

# 5. PCIe by phase (horizontal)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
labels = ["decode","prefill (typical)","prefill burst\n(high context)"]
vals = [0.05, 3.3, 15.5]; cols = [BLUE, AQUA, ORANGE]
yi = np.arange(len(labels))
ax.barh(yi, vals, color=cols, height=0.55, zorder=3)
for i,v in enumerate(vals):
    ax.text(v+0.25, i, ("~0" if v<0.1 else f"{v}")+" GB/s", va="center", fontsize=10, color="#52514e")
ax.axvline(15.75, color="#d03b3b", ls="--", lw=1.2)
ax.text(15.75, -0.75, "Gen3 ×16 ceiling", color="#d03b3b", ha="right", fontsize=9.5)
ax.set_yticks(yi); ax.set_yticklabels(labels); ax.set_xlim(0, 17.5); ax.invert_yaxis()
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.set_title("Intercard PCIe traffic by phase", color=INK, fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax.set_xlabel("GB/s")
save(fig, "05_pcie.jpg")

# 6. n_ubatch
fig, ax = plt.subplots(figsize=(6.4, 3.3))
xi = np.arange(len(UB))
ax.plot(xi, UB98, "-o", color=ORANGE, lw=2, ms=5, mfc=BG, mec=ORANGE, mew=1.6, label="98K context", zorder=3)
ax.plot(xi, UB256, "-o", color=BLUE, lw=2, ms=5, mfc=BG, mec=BLUE, mew=1.6, label="256K context", zorder=3)
ax.set_xticks(xi); ax.set_xticklabels(UB)
ax.set_ylim(700,1300); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.set_title("n_ubatch → prefill throughput (default 512 near-optimal)", color=INK,
             fontsize=12, fontweight="bold", loc="left", pad=10)
ax.set_xlabel("n_ubatch (forward-pass width)"); ax.set_ylabel("prefill tok/s")
ax.legend(frameon=False, fontsize=10)
save(fig, "06_ubatch.jpg")
print("ALL CHARTS DONE ->", OUT)
