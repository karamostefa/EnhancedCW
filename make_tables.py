"""
make_tables.py
==============
Writes the LaTeX tables of the manuscript (tab_*.tex) from the CSV files of ecw_des_experiments.py, so that
every number in a table comes straight from the data.  It also prints a sheet of the numbers that are quoted
in the text.

    python make_tables.py [output_dir]
"""
import csv
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else HERE


def read(name):
    with open(os.path.join(HERE, name), newline="") as fh:
        return list(csv.DictReader(fh))


def by(rows, key):
    return {r[key]: r for r in rows}


def w(name, text):
    with open(os.path.join(OUT, name), "w", newline="\n") as fh:
        fh.write(text)


def sgn(x, nd=1, pct=False):
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0.0:
        return f"{0:.{nd}f}" + ("\\%" if pct else "")
    return ("$+$" if x > 0 else "$-$") + s + ("\\%" if pct else "")


def thousands(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


# ----------------------------------------------------------------------------------------------
# Table 1
# ----------------------------------------------------------------------------------------------
def tab_main():
    rows = read("des_table1.csv")
    names = {
        "PoW": "PoW",
        "PoS": "PoS",
        "PBFT": "PBFT",
        "CW-PoW": "CW-PoW",
        "Enhanced CW-PoW (useful work)": "Enhanced CW-PoW (useful work)",
        "Enhanced CW-PoW (idle waiting)": "Enhanced CW-PoW (idle waiting)",
        "Green-PoW": "Green-PoW$^{\\dagger}$",
        "Proof-of-Experience": "Proof-of-Experience$^{\\dagger}$",
    }
    out = ["\\begin{tabular}{lrrrrrrrr}", "\\hline",
           "\\textbf{Protocol} & \\textbf{$L$ (ms)} & \\textbf{$Th$ (1/s)} & \\textbf{$W$ ($10^3$)} & \\textbf{$E$ (u)} "
           "& \\textbf{$e$ ($10^{-3}$ u)} & \\textbf{$D$} & \\textbf{$M$} & \\textbf{$B$ (kB)} \\\\", "\\hline"]
    for r in rows:
        L = float(r["mean_L"]) * 1e3
        Th = 1.0 / float(r["mean_L"])
        W = float(r["mean_W"])
        E = float(r["mean_E"])
        D = float(r["mean_D"])
        M = float(r["mean_MB"])
        B = float(r["mean_BB"]) / 1e3
        e = (E * 10.0 / W * 1e3) if W > 0 else None
        Ltxt = f"{L:.1f}"
        Wtxt = f"{W / 1e3:.1f}" if W > 0 else "--"
        etxt = f"{e:.3f}" if e is not None else "--"
        Btxt = f"{B:,.1f}".replace(",", "\\,")
        out.append(f"{names[r['protocol']]} & {Ltxt} & {Th:.2f} & {Wtxt} & {E:.2f} & {etxt} & {D:.2f} & {M:.0f} & {Btxt} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_main.tex", "\n".join(out) + "\n")


# ----------------------------------------------------------------------------------------------
# ablation tables
# ----------------------------------------------------------------------------------------------
def g(v, k, kind="mean"):
    return float(v[f"{kind}_{k}"])


def tab_ablation():
    a = by(read("des_ablation.csv"), "variant")

    def line(label, key, ref=None, bold=False):
        v = a[key]
        L, E, M = g(v, "L") * 1e3, g(v, "E"), g(v, "MB")
        if ref is None:
            d1 = d2 = "--"
        else:
            r = a[ref]
            d1 = sgn(L - g(r, "L") * 1e3, 1)
            d2 = sgn(E - g(r, "E"), 2)
        lab = f"\\textbf{{{label}}}" if bold else label
        return f"{lab} & {L:.1f} & {d1} & {E:.2f} & {d2} & {M:.0f} \\\\"

    out = ["\\begin{tabular}{lrrrrr}", "\\hline",
           "\\textbf{Configuration} & \\textbf{$L$ (ms)} & \\textbf{$\\Delta L$ (ms)} & \\textbf{$E$ (units)} & \\textbf{$\\Delta E$ (units)} & \\textbf{$M$} \\\\",
           "\\hline",
           "\\multicolumn{6}{l}{\\emph{Useful-work instantiation (reference: all mechanisms, APT with the original targets)}} \\\\",
           line("Full (APT saturates at $(NbrR,NbrS)=(12,1)$)", "FULL_UW", None, True),
           line("$-$ APT, fixed $(12,1)$", "UW-APT121", "FULL_UW"),
           line("$-$ APT, fixed $(12,6)$", "UW-APT126", "FULL_UW"),
           line("$-$ APT, fixed $(6,3)$", "UW-APT63", "FULL_UW"),
           line("$-$ commit-reveal", "UW-CR", "FULL_UW"),
           line("$-$ pipelining", "UW-PIPE", "FULL_UW"),
           line("$-$ committee finalization", "UW-COM", "FULL_UW"),
           line("$-$ useful work (idle waiting)", "UW-UW", "FULL_UW"),
           "\\hline",
           "\\multicolumn{6}{l}{\\emph{Duty-cycling instantiation (reference: all mechanisms, APT with the original targets)}} \\\\",
           line("Full (APT saturates at $(NbrR,NbrS)=(12,1)$)", "FULL_DC", None, True),
           line("$-$ APT, fixed $(12,1)$", "DC-APT121", "FULL_DC"),
           line("$-$ APT, fixed $(6,3)$", "DC-APT63", "FULL_DC"),
           line("$-$ commit-reveal", "DC-CR", "FULL_DC"),
           line("$-$ pipelining", "DC-PIPE", "FULL_DC"),
           line("$-$ committee finalization", "DC-COM", "FULL_DC"),
           line("$-$ duty-cycling (idle waiting)", "DC-DC", "FULL_DC"),
           "\\hline", "\\end{tabular}"]
    w("tab_ablation_sub.tex", "\n".join(out) + "\n")

    def add(label, key, bold=False):
        v = a[key]
        r = a["B0"]
        L, E, M = g(v, "L") * 1e3, g(v, "E"), g(v, "MB")
        L0, E0 = g(r, "L") * 1e3, g(r, "E")
        lab = f"\\textbf{{{label}}}" if bold else label
        if key == "B0":
            return f"{lab} & {L:.1f} & -- & -- & {E:.2f} & -- & -- & {M:.0f} \\\\"
        return (f"{lab} & {L:.1f} & {sgn(L - L0, 1)} & {sgn(100 * (L - L0) / L0, 1, True)} & {E:.2f} & "
                f"{sgn(E - E0, 2)} & {sgn(100 * (E - E0) / E0, 1, True)} & {M:.0f} \\\\")

    out = ["\\begin{tabular}{lrrrrrrr}", "\\hline",
           "\\textbf{Configuration} & \\textbf{$L$ (ms)} & \\textbf{$\\Delta L$ (ms)} & \\textbf{$\\Delta L$ (\\%)} & \\textbf{$E$ (units)} "
           "& \\textbf{$\\Delta E$ (units)} & \\textbf{$\\Delta E$ (\\%)} & \\textbf{$M$} \\\\", "\\hline",
           add("B0: basic structure, $(6,3)$", "B0", True)]
    for label, key in [("$+$ commit-reveal", "B0+CR"), ("$+$ pipelining (wait overlap)", "B0+PIPE"),
                       ("$+$ pipelining (hash overlap)", "B0+PIPEH"),
                       ("$+$ committee finalization", "B0+COM"), ("$+$ useful work", "B0+UW"),
                       ("$+$ duty-cycling", "B0+DC"),
                       ("$+$ APT (original targets, saturates)", "B0+APT"),
                       ("All mechanisms except APT, idle waiting", "FULL_IDLE63"),
                       ("All mechanisms except APT, duty-cycling", "DC-APT63"),
                       ("All mechanisms except APT, useful work", "UW-APT63")]:
        out.append(add(label, key))
    out += ["\\hline", "\\end{tabular}"]
    w("tab_ablation_add.tex", "\n".join(out) + "\n")


def tab_robust():
    sets = [("baseline", "des_ablation.csv"), ("$d=2$\\,ms", "des_ablation_d2ms.csv"),
            ("$d=10$\\,ms", "des_ablation_d10ms.csv"), ("$t_{round}=100$\\,ms", "des_ablation_t100ms.csv")]
    data = [(n, by(read(f), "variant")) for n, f in sets]
    mech = [("commit-reveal", "B0+CR"), ("pipelining (wait overlap)", "B0+PIPE"), ("pipelining (hash overlap)", "B0+PIPEH"),
            ("committee finalization", "B0+COM"),
            ("useful work", "B0+UW"), ("duty-cycling", "B0+DC")]
    out = ["\\begin{tabular}{l" + "r" * len(sets) + "}", "\\hline",
           "\\textbf{Mechanism added to B0} & " + " & ".join(f"\\textbf{{{n}}}" for n, _ in sets) + " \\\\", "\\hline"]
    for lab, key in mech:
        cells = []
        for n, d in data:
            L0, E0 = g(d["B0"], "L"), g(d["B0"], "E")
            L, E = g(d[key], "L"), g(d[key], "E")
            cells.append(f"{sgn(100 * (E - E0) / E0, 1)} / {sgn(100 * (L - L0) / L0, 1)}")
        out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    cells = []
    for n, d in data:
        L0, E0 = g(d["B0"], "L"), g(d["B0"], "E")
        L, E = g(d["POW_RACE"], "L"), g(d["POW_RACE"], "E")
        cells.append(f"{sgn(100 * (E - E0) / E0, 1)} / {sgn(100 * (L - L0) / L0, 1)}")
    out.append("PoW race of equal work (instead of B0) & " + " & ".join(cells) + " \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_robust.tex", "\n".join(out) + "\n")


# ----------------------------------------------------------------------------------------------
# other tables
# ----------------------------------------------------------------------------------------------
def tab_wake():
    rows = read("des_wake_sweep.csv")
    out = ["\\begin{tabular}{rrrrr}", "\\hline",
           "\\textbf{$E_{wake}$ (units)} & \\textbf{$E$ idle (units)} & \\textbf{$E$ sleep (units)} & \\textbf{$\\Delta E$ (\\%)} & \\textbf{waste change (\\%)} \\\\",
           "\\hline"]
    for r in rows:
        ew = float(r["E_wake"])
        if ew not in (0.0, 0.01, 0.05, 0.1, 0.3, 1.0):
            continue
        out.append(f"{ew:g} & {float(r['E_idle']):.3f} & {float(r['E_sleep']):.3f} & {sgn(float(r['dE_pct']), 2, True)} & {sgn(float(r['dwaste_pct']), 0, True)} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_wake.tex", "\n".join(out) + "\n")


def tab_withhold():
    rows = read("des_withholding.csv")
    out = ["\\begin{tabular}{rrrrr}", "\\hline",
           "\\textbf{$\\tau_{rev}$ (ms)} & \\textbf{$L$ without (ms)} & \\textbf{$L$ with $\\ge 1$ (ms)} & \\textbf{increase (ms)} & \\textbf{$(R-1)(\\tau_{rev}-d_{rev})$ (ms)} \\\\",
           "\\hline"]
    for tau in sorted({float(r["timeout_s"]) for r in rows}):
        rr = [r for r in rows if abs(float(r["timeout_s"]) - tau) < 1e-12]
        L0 = float([r for r in rr if int(r["withholders"]) == 0][0]["L_s"]) * 1e3
        Ls = [float(r["L_s"]) * 1e3 for r in rr if int(r["withholders"]) > 0]
        L1 = sum(Ls) / len(Ls)
        out.append(f"{tau * 1e3:g} & {L0:.1f} & {L1:.1f} & {L1 - L0:.1f} & {5 * (tau - 0.001) * 1e3:.1f} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_withhold.tex", "\n".join(out) + "\n")


def tab_spikes():
    rows = read("des_spikes.csv")
    out = ["\\begin{tabular}{rrrrr}", "\\hline",
           "\\textbf{Spike prob.} & \\textbf{Spike duration (ms)} & \\textbf{$L$ (ms)} & \\textbf{$\\Delta L$ (\\%)} & \\textbf{$E$ (units)} \\\\", "\\hline"]
    L0 = float(rows[0]["L_s"])
    for r in rows:
        L = float(r["L_s"])
        out.append(f"{float(r['spike_prob']):.1f} & {float(r['spike_dur_s']) * 1e3:.0f} & {L * 1e3:.1f} & "
                   f"{sgn(100 * (L - L0) / L0, 1, True) if r is not rows[0] else '--'} & {float(r['E']):.2f} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_spikes.tex", "\n".join(out) + "\n")


def tab_saturation():
    rows = read("des_apt_saturation.csv")
    out = ["\\begin{tabular}{ccrrrr}", "\\hline",
           "\\textbf{$(NbrR_0,NbrS_0)$} & \\textbf{Final $(NbrR,NbrS)$} & \\textbf{Block of $NbrR=12$} & \\textbf{Block of $NbrS=1$} & \\textbf{$L$, block 8 (ms)} & \\textbf{$E$, block 8 (units)} \\\\", "\\hline"]
    for r in rows:
        out.append(f"({int(r['R0'])},{int(r['S0'])}) & ({float(r['R_final']):.0f},{float(r['S_final']):.0f}) & {int(r['block_R12'])} & {(str(int(r['block_S1'])) if int(r['block_S1']) > 0 else '--')} & "
                   f"{float(r['L_block8_s']) * 1e3:.0f} & {float(r['E_block8']):.1f} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_saturation.tex", "\n".join(out) + "\n")


def tab_crosscheck():
    """Prototype (multi-threaded, 5 blocks) versus event-driven model: percent change when a mechanism is added to B0."""
    proto = by(read("ablation_mechanisms_summary.csv"), "variant")
    model = by(read("des_ablation.csv"), "variant")
    pairs = [("commit-reveal", "B0+CR", "B0+CR"), ("pipelining (wait overlap)", "B0+PIPE", "B0+PIPE"),
             ("committee finalization", "B0+COMMITTEE", "B0+COM"), ("useful work", "B0+UW", "B0+UW"),
             ("duty-cycling ($E_{wake}=1$)", "B0+DC", "B0+DC"), ("APT, original targets", "B0+APT", "B0+APT")]
    out = ["\\begin{tabular}{lrrrr}", "\\hline",
           " & \\multicolumn{2}{c}{\\textbf{Prototype}} & \\multicolumn{2}{c}{\\textbf{Event-driven model}} \\\\",
           "\\textbf{Mechanism added to B0} & \\textbf{$\\Delta E$ (\\%)} & \\textbf{$\\Delta L$ (\\%)} & \\textbf{$\\Delta E$ (\\%)} & \\textbf{$\\Delta L$ (\\%)} \\\\",
           "\\hline"]
    E0p, L0p = float(proto["B0"]["mean_E_block"]), float(proto["B0"]["mean_L"])
    E0m, L0m = g(model["B0"], "E"), g(model["B0"], "L")
    for lab, kp, km in pairs:
        pe = 100 * (float(proto[kp]["mean_E_block"]) / E0p - 1)
        pl = 100 * (float(proto[kp]["mean_L"]) / L0p - 1)
        me = 100 * (g(model[km], "E") / E0m - 1)
        ml = 100 * (g(model[km], "L") / L0m - 1)
        out.append(f"{lab} & {sgn(pe, 0)} & {sgn(pl, 0)} & {sgn(me, 1)} & {sgn(ml, 1)} \\\\")
        print(f"CROSSCHECK {lab:32s} proto dE {pe:+7.1f}% dL {pl:+7.1f}%   model dE {me:+7.1f}% dL {ml:+7.1f}%")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_crosscheck.tex", "\n".join(out) + "\n")


def tab_equal():
    rows = read("des_equal.csv")
    out = ["\\begin{tabular}{rrrrrrrrrrr}", "\\hline",
           " & & & \\multicolumn{3}{c}{\\textbf{CW-PoW}} & \\multicolumn{2}{c}{\\textbf{PoW, equal mean time}} & \\multicolumn{2}{c}{\\textbf{Difference}} & \\\\",
           "\\textbf{$N_m$} & \\textbf{$S$} & \\textbf{$d$ (ms)} & \\textbf{$L$ (ms)} & \\textbf{$L_{99}$ (ms)} & \\textbf{$E$ (u)} "
           "& \\textbf{$L_{99}$ (ms)} & \\textbf{$E$ (u)} & \\textbf{$E$} & \\textbf{$L_{99}$} & \\textbf{$M$} \\\\", "\\hline"]
    prev = None
    for r in rows:
        n = int(r["N"])
        if int(r["S"]) >= n:
            continue                         # S = N_m is outside the design of the protocol (S is a constant of the order of ten)
        if prev is not None and n != prev:
            out.append("\\hline")
        prev = n
        out.append(f"{n} & {int(r['S'])} & {float(r['d_s']) * 1e3:.0f} & {float(r['L_mean_s']) * 1e3:.0f} & {float(r['L_p99_s']) * 1e3:.0f} & {float(r['E']):.2f} & "
                   f"{float(r['pow_p99_s']) * 1e3:.0f} & {float(r['pow_E']):.1f} & {sgn(float(r['dE_pct']), 0, True)} & {sgn(float(r['dp99_pct']), 0, True)} & {float(r['MB']):.0f} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_equal.tex", "\n".join(out) + "\n")


def tab_fairness():
    rows = read("des_fairness.csv")
    gcols = [f"g{i}" for i in range(1, 11)]

    def lo_hi(r):
        v = [float(r[c]) for c in gcols if r[c] != ""]
        return v[0], v[-1]
    entry = {(int(r["N"]), r["S"].replace("_entry", "")): r for r in rows if str(r["S"]).endswith("_entry")}
    out = ["\\begin{tabular}{rrrrrrrrr}", "\\hline",
           " & & & & & \\multicolumn{2}{c}{\\textbf{$F$ of the slowest / fastest group}} & \\multicolumn{2}{c}{\\textbf{Admission ratio}} \\\\",
           "\\textbf{$N_m$} & \\textbf{$S$} & \\textbf{Blocks} & \\textbf{Jain} & \\textbf{$F_{\\min}$--$F_{\\max}$} & \\textbf{slowest} & \\textbf{fastest} & \\textbf{slowest} & \\textbf{fastest} \\\\", "\\hline"]
    prev = None
    for r in rows:
        if str(r["S"]).endswith("_entry") or str(r["S"]).endswith("_first"):
            continue
        n = int(r["N"])
        if prev is not None and n != prev:
            out.append("\\hline")
        prev = n
        lo, hi = lo_hi(r)
        if r["S"] not in ("reference",) and int(r["S"]) >= n:
            continue
        if r["S"] == "reference":
            out.append(f"{n} & ideal & {int(r['blocks'])} & {float(r['jain']):.3f} & {float(r['F_min']):.2f}--{float(r['F_max']):.2f} & {lo:.2f} & {hi:.2f} & -- & -- \\\\")
            continue
        e = entry[(n, r["S"])]
        elo, ehi = lo_hi(e)
        out.append(f"{n} & {int(r['S'])} & {int(r['blocks'])} & {float(r['jain']):.3f} & {float(r['F_min']):.2f}--{float(r['F_max']):.2f} & {lo:.2f} & {hi:.2f} & {elo:.2f} & {ehi:.2f} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    w("tab_fairness.tex", "\n".join(out) + "\n")


def sheet():
    print("=" * 100)
    a = by(read("des_ablation.csv"), "variant")
    t = by(read("des_table1.csv"), "protocol")
    b0, pw = a["B0"], a["POW_RACE"]
    print(f"B0: L={g(b0, 'L') * 1e3:.1f} E={g(b0, 'E'):.2f} (hash {g(b0, 'E_hash'):.2f} ovh {g(b0, 'E_ovh'):.2f} waste {g(b0, 'E_waste'):.2f}) D={g(b0, 'D'):.3f}")
    print(f"PoW race equal work: L={g(pw, 'L') * 1e3:.1f} E={g(pw, 'E'):.2f}  -> B0/PoW: L {100 * (g(b0, 'L') / g(pw, 'L') - 1):+.1f}%  E {100 * (g(b0, 'E') / g(pw, 'E') - 1):+.1f}%")
    for k in ("FULL_UW", "FULL_UW_CAL", "FULL_DC", "FULL_IDLE", "FULL_IDLE63", "UW-APT63", "DC-APT63", "UW+PIPEH",
              "B0+CR", "B0+PIPE", "B0+COM", "B0+UW", "B0+DC", "B0+APT"):
        if k in a:
            v = a[k]
            print(f"{k:12s} L={g(v, 'L') * 1e3:7.1f} (ci {g(v, 'L', 'ci95') * 1e3:.1f}) E={g(v, 'E'):6.2f} (ci {g(v, 'E', 'ci95'):.2f}) "
                  f"[h {g(v, 'E_hash'):.1f} o {g(v, 'E_ovh'):.2f} u {g(v, 'E_uw'):.2f} w {g(v, 'E_waste'):.2f}] D={g(v, 'D'):.3f} M={g(v, 'MB'):.0f} R={g(v, 'R'):.2f} "
                  f"| dL vs B0 {100 * (g(v, 'L') / g(b0, 'L') - 1):+.1f}%  dE vs B0 {100 * (g(v, 'E') / g(b0, 'E') - 1):+.1f}%")
    for r in read("des_table1.csv"):
        L = float(r["mean_L"])
        E = float(r["mean_E"])
        W = float(r["mean_W"])
        print(f"T1 {r['protocol']:32s} L={L * 1e3:8.1f} Th={1 / L:6.2f} W={W / 1e3:6.1f}k E={E:6.2f} e={(E * 10 / W * 1e3 if W else float('nan')):.3f} D={float(r['mean_D']):.3f}")


if __name__ == "__main__":
    tab_main()
    tab_ablation()
    tab_robust()
    tab_wake()
    tab_withhold()
    tab_spikes()
    tab_saturation()
    tab_crosscheck()
    tab_equal()
    tab_fairness()
    sheet()
    print("tables written to", OUT)
