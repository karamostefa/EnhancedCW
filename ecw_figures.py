"""
ecw_figures.py
==============
Regenerates every figure of the manuscript from the CSV files written by ecw_des_experiments.py.
Common style: serif font (Palatino Linotype, as in the MDPI template), decimal points, 8 pt text,
one colour per protocol, 95 % confidence intervals where they are visible.

    python ecw_figures.py            ->  img/fig_*.pdf
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Book Antiqua", "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic",
    "mathtext.bf": "Palatino Linotype:bold",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
})

BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#1f4e9c", "#b22222", "#2e7d32", "#e08a00", "#6b6b6b", "#6a3d9a"
P_SLEEP_OVER_P_IDLE = 0.5 / 5.0
W2 = 6.3            # full text width (inches)


def read(name):
    with open(os.path.join(HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


def f(rows, key):
    return np.array([float(r[key]) for r in rows])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    if os.environ.get('ECW_PREVIEW'):
        pv = os.environ['ECW_PREVIEW']
        os.makedirs(pv, exist_ok=True)
        fig.savefig(os.path.join(pv, name + '.png'), dpi=170)


def panel_label(ax, s):
    ax.text(-0.17, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------------------------
# energy model: relative energy versus duty cycle, idle share
# ---------------------------------------------------------------------------------------------
def fig_energy_model():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4))
    D = np.linspace(0, 1, 400)
    D0 = float({r["variant"]: r for r in read("des_ablation.csv")}["B0"]["mean_D"])   # duty cycle of the basic structure
    ax = axs[0]
    for rho, c, ls in [(1, GREY, ":"), (5, GREEN, "--"), (20, BLUE, "-"), (200, RED, "-.")]:
        ax.plot(D, D + (1 - D) / rho, color=c, ls=ls, label=r"$\rho=%d$" % rho)
    ax.plot(D0, D0 + (1 - D0) / 20, "o", color=BLUE, mfc="white")
    ax.annotate("basic structure\n$D=%.2f$" % D0, (D0, D0 + (1 - D0) / 20), xytext=(0.04, 0.62), fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
    ax.set_xlabel(r"Duty cycle $D$ (dimensionless)")
    ax.set_ylabel(r"$E/(P_a T)$ (dimensionless)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "(a)")
    ax = axs[1]
    for rho, c, ls in [(5, GREEN, "--"), (20, BLUE, "-"), (200, RED, "-.")]:
        ax.plot(D, 100 * (1 - D) / (rho * D + 1 - D), color=c, ls=ls, label=r"$\rho=%d$" % rho)
    ax.plot(D0, 100 * (1 - D0) / (20 * D0 + 1 - D0), "o", color=BLUE, mfc="white")
    ax.set_xlabel(r"Duty cycle $D$ (dimensionless)")
    ax.set_ylabel(r"Idle share $\varepsilon(D)$ of the energy (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_energy_model")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# round timeline (specification of pipelining)
# ---------------------------------------------------------------------------------------------
def fig_timeline():
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 4.55)
    ax.axis("off")
    cols = {"hdr": ORANGE, "hash": BLUE, "wait": "#d9d9d9", "rev": GREEN, "pre": "#f3c677", "fin": ORANGE}

    def seg(y, x0, w, key, label=None, tc="white"):
        ax.add_patch(Rectangle((x0, y), w, 0.62, fc=cols[key], ec="black", lw=0.5))
        if label:
            ax.text(x0 + w / 2, y + 0.31, label, ha="center", va="center", fontsize=6.8, color=tc)

    rows = [("(a) no pipelining", 3.45), ("(b) wait overlap", 2.15), ("(c) hash overlap", 0.85)]
    for name, y in rows:
        ax.text(0, y + 0.72, name, fontsize=8, fontweight="bold", va="bottom")
    # (a)
    y = 3.45
    seg(y, 0, 30, "hash", "hashing, round $r$")
    seg(y, 30, 12, "wait", "wait", tc="black")
    seg(y, 42, 5, "rev", "$d_{rev}$")
    seg(y, 47, 14, "hdr", "header $t_{hdr}$")
    seg(y, 61, 30, "hash", "hashing, round $r+1$")
    ax.annotate("", xy=(61, y + 1.0), xytext=(50, y + 1.0), arrowprops=dict(arrowstyle="<->", lw=0.6, color=RED))
    ax.text(55.5, y + 1.04, "saving", ha="center", va="bottom", fontsize=6.8, color=RED)
    # (b)
    y = 2.15
    seg(y, 0, 30, "hash", "hashing, round $r$")
    seg(y, 30, 11, "pre", "precompute", tc="black")
    seg(y, 41, 1.0, "wait")
    seg(y, 42, 5, "rev", "$d_{rev}$")
    seg(y, 47, 3, "fin", "")
    seg(y, 50, 30, "hash", "hashing, round $r+1$")
    ax.annotate("", xy=(30, y + 0.74), xytext=(42, y + 0.74), arrowprops=dict(arrowstyle="<->", lw=0.5))
    ax.text(36, y + 0.78, "wait $w_i$", ha="center", va="bottom", fontsize=6.8)
    ax.text(48.5, y - 0.1, r"$\varphi t_{hdr}$", ha="center", va="top", fontsize=7)
    # (c)
    y = 0.85
    seg(y, 0, 30, "hash", "hashing, round $r$")
    ax.add_patch(Rectangle((4, y - 0.3), 12, 0.18, fc=cols["pre"], ec="black", lw=0.5))
    ax.text(17.5, y - 0.21, r"precompute $(1-\varphi)t_{hdr}$ (separate unit)", fontsize=6.0, va="center")
    seg(y, 30, 12, "wait", "wait", tc="black")
    seg(y, 42, 5, "rev", "$d_{rev}$")
    seg(y, 47, 3, "fin", "")
    seg(y, 50, 30, "hash", "hashing, round $r+1$")
    ax.text(48.5, y - 0.1, r"$\varphi t_{hdr}$", ha="center", va="top", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_timeline.pdf"))
    if os.environ.get("ECW_PREVIEW"):
        fig.savefig(os.path.join(os.environ["ECW_PREVIEW"], "fig_timeline.png"), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# scalability
# ---------------------------------------------------------------------------------------------
def fig_scalability():
    rows = read("des_scalability.csv")
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.5))
    spec = [("mean_L", "ci95_L", 1e3, "Block latency $L$ (ms)", False),
            ("mean_E", "ci95_E", 1.0, "Energy $E$ (units)", False),
            ("mean_MB", "ci95_MB", 1.0, "Messages per block $M$", True),
            ("mean_BB", "ci95_BB", 1e-3, "Data per block $B$ (kB)", True)]
    for ax, (key, cik, sc, lab, logy), tag in zip(axs.ravel(), spec, "abcd"):
        for proto, c, mk in [("CW-PoW", BLUE, "o"), ("Enhanced CW-PoW", RED, "s")]:
            rr = [r for r in rows if r["protocol"] == proto]
            N = f(rr, "N")
            y = f(rr, key) * sc
            e = f(rr, cik) * sc
            ax.errorbar(N, y, yerr=e, color=c, marker=mk, capsize=2, label=proto)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([10, 50, 100, 500, 1000])
        ax.set_xticklabels(["10", "50", "100", "500", "1000"])
        ax.set_xlabel("Number of miners $N$")
        ax.set_ylabel(lab)
        panel_label(ax, "(%s)" % tag)
        if tag == "c":
            ax.legend(frameon=False, loc="upper left")
        if not logy:
            ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    save(fig, "fig_scalability")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# ablation: energy decomposition and latency
# ---------------------------------------------------------------------------------------------
def fig_ablation():
    rows = {r["variant"]: r for r in read("des_ablation.csv")}

    def val(v, k):
        return float(rows[v]["mean_" + k])

    spec = [
        ("PoW race, equal work", "POW_RACE", 0),
        ("B0: basic structure", "B0", 0),
        ("Full, idle waiting", "FULL_IDLE63", 0),
        ("Full, duty-cycling", "DC-APT63", 0),
        ("Full, useful work", "UW-APT63", 0),
        ("APT, first rule (saturates)", "FULL_UW", 1),
        ("APT, latency priority ($L^{\\ast}=0.25$ s)", "FULL_UW_CAL", 1),
    ]
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ypos, y = [], 0.0
    prev = 0
    comps = [("E_hash", "Hashing", BLUE), ("E_ovh", "Protocol overhead", ORANGE), ("E_uw", "Useful work", GREEN),
             ("E_waste", "Idle, sleep, wake-up", GREY)]
    labels_done = set()
    for lab, key, grp in spec:
        if grp != prev:
            y += 0.55
            prev = grp
        left = 0.0
        for ck, cl, cc in comps:
            v = val(key, ck) if key != "POW_RACE" else (val(key, "E") if ck == "E_hash" else 0.0)
            ax.barh(y, v, left=left, color=cc, edgecolor="white", linewidth=0.4,
                    label=cl if cl not in labels_done else None, height=0.62)
            labels_done.add(cl)
            left += v
        L = val(key, "L") * 1e3
        R = val(key, "R") if key != "POW_RACE" else float("nan")
        ax.text(left + 0.6, y, "$E$=%.1f,  $L$=%.0f ms" % (val(key, "E"), L), va="center", fontsize=7)
        ypos.append((y, lab))
        y += 1.0
    ax.set_yticks([p for p, _ in ypos])
    ax.set_yticklabels([l for _, l in ypos])
    ax.invert_yaxis()
    ax.set_xlabel("Energy per miner and block $E$ (units)")
    ax.set_xlim(0, 1.45 * max(val(k, 'E') for _, k, _ in spec))
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.56), ncol=1)
    ax.text(ax.get_xlim()[1] * 0.99, -0.12, "fixed $(R,S)=(6,3)$ in the first five rows", ha="right", va="center", fontsize=6.8, color=GREY)
    fig.tight_layout()
    save(fig, "fig_ablation")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# APT
# ---------------------------------------------------------------------------------------------
def fig_apt():
    sweep = read("des_apt_target_sweep.csv")
    reg = read("des_apt_regime.csv")
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.5))
    for rule, c, mk, lab in [("original", RED, "s", "independent rules"), ("priority", BLUE, "o", "latency priority")]:
        rr = [r for r in sweep if r["rule"] == rule]
        t = f(rr, "target_latency_s") * 1e3
        axs[0, 0].plot(t, f(rr, "NbrR_mean"), color=c, marker=mk, label=lab)
        axs[0, 1].plot(t, f(rr, "L_mean_s") * 1e3, color=c, marker=mk, label=lab)
    axs[0, 0].axhline(12, color=GREY, ls=":", lw=0.8)
    axs[0, 0].text(105, 12.35, r"$R_{\max}$", ha="left", fontsize=7, color=GREY)
    axs[0, 0].set_xscale("log")
    axs[0, 1].set_xscale("log")
    lo, hi = 100, 1000
    axs[0, 1].plot([lo, hi], [lo, hi], color=GREY, ls=":", lw=0.8)
    axs[0, 1].text(560, 640, "$L=L^{\\ast}$", fontsize=7, color=GREY, rotation=0)
    for ax in axs[0]:
        ax.set_xlabel("Latency target $L^{\\ast}$ (ms)")
        ax.set_xticks([100, 200, 500, 1000])
        ax.set_xticklabels(["100", "200", "500", "1000"])
    axs[0, 0].set_ylabel("Steady-state $NbrR$")
    axs[0, 1].set_ylabel("Steady-state latency $L$ (ms)")
    axs[0, 0].set_ylim(0, 13.5)
    axs[0, 0].legend(frameon=False, loc="lower right")
    panel_label(axs[0, 0], "(a)")
    panel_label(axs[0, 1], "(b)")
    cases = [("APT, original rule", RED, "-"), ("APT, priority rule", BLUE, "-"), ("static (6,3)", GREEN, "--"),
             ("static (12,6)", GREY, ":")]
    labmap = {"APT, original rule": "APT, independent rules", "APT, priority rule": "APT, latency priority",
              "static (6,3)": "static (6,3)", "static (12,6)": "static (12,6)"}
    for case, c, ls in cases:
        rr = [r for r in reg if r["case"] == case]
        b = f(rr, "block")
        axs[1, 0].plot(b, f(rr, "NbrR_mean"), color=c, ls=ls, label=labmap[case])
        axs[1, 1].plot(b, f(rr, "L_mean_s") * 1e3, color=c, ls=ls, label=labmap[case])
    for ax in axs[1]:
        ax.axvspan(21, 40.5, color="#f2d6d6", alpha=0.6, lw=0)
        ax.set_xlabel("Block")
        ax.set_xlim(1, 60)
    axs[1, 1].set_ylim(100, 820)
    axs[1, 1].axhline(250, color="black", lw=0.6, ls=":")
    axs[1, 1].text(59, 150, "target $L^{\\ast}$=250 ms (dotted)", ha="right", fontsize=7)
    axs[1, 0].set_ylabel("$NbrR$ (mean of the repetitions)")
    axs[1, 1].set_ylabel("Block latency $L$ (ms)")
    axs[1, 0].text(30.7, 11.2, "+25 ms delay", ha="center", fontsize=7, color=RED)
    axs[1, 1].legend(frameon=False, loc="upper left", fontsize=7)
    panel_label(axs[1, 0], "(c)")
    panel_label(axs[1, 1], "(d)")
    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    save(fig, "fig_apt")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# pipelining and duty cycling
# ---------------------------------------------------------------------------------------------
def fig_pipe_dc():
    pipe = read("des_pipelining.csv")
    dc = read("des_dc_roundlength.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    for d, c, mk in [(0.0, BLUE, "o"), (0.005, GREEN, "s"), (0.020, ORANGE, "^")]:
        rr = [r for r in pipe if abs(float(r["net_delay_s"]) - d) < 1e-9]
        ax.plot(f(rr, "t_hdr_s") * 1e3, f(rr, "saving_wait_s") * 1e3, color=c, marker=mk, label="wait overlap, $d$=%g ms" % (d * 1e3))
    rr = [r for r in pipe if abs(float(r["net_delay_s"])) < 1e-9]
    ax.plot(f(rr, "t_hdr_s") * 1e3, f(rr, "saving_hash_s") * 1e3, color=RED, lw=1.8, label="hash overlap (= bound)")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Header time $t_{hdr}$ (ms)")
    ax.set_ylabel("Latency saving (ms)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.6, columnspacing=1.2)
    panel_label(ax, "(a)")
    ax = axs[1]
    tr = f(dc, "t_round_s") * 1e3
    ax.plot(tr, f(dc, "dE_sleep_pct"), color=BLUE, marker="o", label=r"duty-cycling, $E_{wake}$=1")
    ax.axhline(0, color="black", lw=0.6)
    wk = read("des_wake_sweep.csv")[0]
    bound = -100.0 * (1 - P_SLEEP_OVER_P_IDLE) * float(wk["waste_idle"]) / float(wk["E_idle"])
    ax.axhline(bound, color=GREY, ls=":", lw=0.9)
    ax.text(5.5, bound - 0.9, r"bound $\varepsilon(D)(1-P_s/P_w)$", fontsize=6.8, color=GREY, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Mean compute time of a round (ms)")
    ax.set_ylabel("Energy change (%)")
    ax.set_ylim(-20, 34)
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "(b)")
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig_pipe_dc")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# equal mean block time, fairness
# ---------------------------------------------------------------------------------------------
def fig_equal():
    rows = [r for r in read("des_equal.csv") if int(r["S"]) < int(r["N"])]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 3.0))
    cols = {3: BLUE, 5: GREEN, 10: ORANGE}
    for S, c in cols.items():
        for d, ls, mk in ((0.0, "-", "o"), (0.02, "--", "s")):
            rr = [r for r in rows if int(r["S"]) == S and abs(float(r["d_s"]) - d) < 1e-9]
            N = f(rr, "N")
            lab = "$S=%d$, $d=%d$ ms" % (S, round(d * 1e3))
            axs[0].plot(N, f(rr, "E"), ls, color=c, marker=mk, label=lab)
            axs[1].plot(N, f(rr, "L_p99_s") * 1e3, ls, color=c, marker=mk)
            axs[2].plot(N, f(rr, "L_mean_s") * 1e3, ls, color=c, marker=mk)
    rr = [r for r in rows if int(r["S"]) == 3 and abs(float(r["d_s"])) < 1e-9]
    axs[0].plot(f(rr, "N"), f(rr, "pow_E"), ":", color=RED, marker="^", label="PoW, equal mean time ($S=3$, $d=0$)")
    axs[1].plot(f(rr, "N"), f(rr, "pow_p99_s") * 1e3, ":", color=RED, marker="^")
    for ax, lab, tag in zip(axs, ["Energy $E$ (units)", "99th percentile of $L$ (ms)", "Mean block time $L$ (ms)"], "abc"):
        ax.set_xscale("log")
        ax.set_xticks([10, 100, 1000])
        ax.set_xticklabels(["10", "100", "1000"])
        ax.set_xlabel("Number of miners $N$")
        ax.set_ylabel(lab)
        panel_label(ax, "(%s)" % tag)
    axs[0].set_yscale("log")
    axs[0].set_ylim(0.7, 50)
    for ax in axs[1:]:
        ax.set_ylim(0, None)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.6, rect=(0, 0.17, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.8)
    save(fig, "fig_equal")
    plt.close(fig)


def fig_fairness():
    rows = read("des_fairness.csv")
    gcols = ["g%d" % i for i in range(1, 11)]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    palette = [BLUE, GREEN, ORANGE, PURPLE, RED, GREY]
    for ax, n, tag in zip(axs, (100, 1000), "ab"):
        k = 0
        for r in rows:
            if int(r["N"]) != n or str(r["S"]).endswith("_entry") or str(r["S"]).endswith("_first") or r["S"] == "reference":
                continue
            S = int(r["S"])
            if S == n:
                continue                       # S = N_m is outside the design of the protocol
            y = [float(r[c]) for c in gcols if r[c] != ""]
            ax.plot(range(1, len(y) + 1), y, marker="o", color=palette[k % len(palette)], label="$S=%d$" % S)
            k += 1
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Hash-power decile of the miners (1 = slowest)")
        ax.set_ylabel("Fairness ratio $F$ of the decile")
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0.0, 1.8)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower right")
        panel_label(ax, "(%s) $N_m=%d$" % (tag, n))
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_fairness")
    plt.close(fig)


if __name__ == "__main__":
    fig_energy_model()
    fig_timeline()
    fig_scalability()
    fig_ablation()
    fig_apt()
    fig_pipe_dc()
    fig_equal()
    fig_fairness()
    print("figures written to", OUT)
