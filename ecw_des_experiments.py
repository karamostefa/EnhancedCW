"""
ecw_des_experiments.py
======================
All numerical experiments of Sections 5.5-5.10 of the manuscript that are based on the event-driven
model of ecw_des.py.  One command regenerates every CSV file used for the tables and figures:

    python ecw_des_experiments.py [reps] [validate|table1|ablation|apt|pipe|dc|scal|byz|all]

Everything is seeded; two runs on any machine give identical numbers.
Output files (written next to this script): des_*.csv
"""
import sys
import time
import math
from dataclasses import replace

import numpy as np

import ecw_des as m
from ecw_des import (Cfg, basic_cfg, run, summarize, replicate, write_csv, KEYS, TAU, P_HIT_ENH, P_HIT_BASIC,
                     P_ACTIVE, P_IDLE, P_SLEEP, E_WAKE, WAKE_THRESHOLD)


def ref_basic(n=10, blocks=20, **kw):
    """Basic CW-PoW as configured in the prototype: 5 rounds, quorum 3, harder per-round target (7 282 hashes)."""
    return basic_cfg(n=n, blocks=blocks, R0=5, S0=3, p_hit=P_HIT_BASIC, t_round=TAU / P_HIT_BASIC, **kw)


def ref_enhanced(n=10, blocks=20, **kw):
    """Enhanced CW-PoW as configured in the prototype: all mechanisms, useful work, APT from (6,3)."""
    return replace(Cfg(), n=n, blocks=blocks, **kw)


def ci(s, k):
    return s[k][2]


# ------------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------------
def exp_validate(reps=400):
    print("VALIDATION of the event-driven model", flush=True)
    c = basic_cfg(blocks=1, R0=6, S0=3)
    assert run(c, 1)[0]["MB"] == 36
    assert run(replace(c, use_cr=True), 1)[0]["MB"] == 61
    assert run(replace(c, wait_mode="uw"), 1)[0]["MB"] == 51
    assert run(replace(c, use_committee=True), 1)[0]["MB"] == 43
    print("  message counts 36 / 61 / 51 / 43 = counts of the multi-threaded prototype")
    assert m.pos_blocks(1)[0]["MB"] == 12 and m.pos_blocks(1)[0]["BB"] == 4200
    assert m.pbft_blocks(1)[0]["MB"] == 27 and m.pbft_blocks(1)[0]["BB"] == 20500
    cb = ref_basic(blocks=1)
    r = run(cb, 1)[0]
    assert r["MB"] == 30 and r["BB"] == 21384
    print("  PoS 12 msgs / 4 200 B, PBFT 27 msgs / 20 500 B, basic CW-PoW 30 msgs / 21 384 B = prototype values")
    # independent Monte Carlo of the protocol rule (plain numpy, no accounting code shared with the model)
    def mc_block(n, S, R, rng, t_round=0.020):
        alive = n
        hash_e = 0.0
        for r in range(R):
            k = min(S, alive)
            x = np.sort(rng.exponential(t_round, alive))
            q = x[k - 1]
            hash_e += (np.minimum(x, q).sum() if r == 0 else np.minimum(x, q).sum()) 
            alive = k
        return hash_e / n
    c = basic_cfg(blocks=1, R0=6, S0=3, hdr_cost=0.0)
    es = [run(c, 5000 + k)[0]["E_hash"] for k in range(reps)]
    rng = np.random.default_rng(1)
    mc = [P_ACTIVE * mc_block(10, 3, 6, rng) for _ in range(40000)]
    print(f"  hashing energy / miner / block: simulated {np.mean(es):.3f}   Monte Carlo of the rule {np.mean(mc):.3f}")
    assert abs(np.mean(es) / np.mean(mc) - 1) < 0.03
    cc = basic_cfg(blocks=1, R0=2, S0=3, hdr_cost=0.0)
    ws = [run(cc, 7000 + k)[0]["wait_mean"] for k in range(reps * 2)]
    rng = np.random.default_rng(1)
    mc = []
    for _ in range(40000):
        x = np.sort(rng.exponential(cc.t_round, 10))[:3]
        mc.append((x[2] - x).mean())
    print(f"  mean quorum wait / surviving miner / round: model {np.mean(ws) * 1e3:.3f} ms   order-statistics Monte Carlo {np.mean(mc) * 1e3:.3f} ms")
    assert abs(np.mean(ws) / np.mean(mc) - 1) < 0.05
    W = 10 * 6 / P_HIT_ENH
    rng = np.random.default_rng(3)
    Es = [m.pow_block(10, W, TAU, 0.0, rng)["E"] for _ in range(20000)]
    print(f"  PoW race at equal expected work: E = {np.mean(Es):.3f}   analytic P_a*R*t_round = {6 * 0.020 * 100:.3f}")
    assert abs(np.mean(Es) / (6 * 0.020 * 100) - 1) < 0.02
    rng = np.random.default_rng(5)
    Lp = [m.pow_blocks(10, 1, rng)[0]["L"] for _ in range(20000)]
    print(f"  PoW block time at the prototype target: {np.mean(Lp) * 1e3:.1f} ms   (prototype, 20-block matched run: 348 ms)")
    print("  all checks passed", flush=True)


# ------------------------------------------------------------------------------------------------
# Table 1
# ------------------------------------------------------------------------------------------------
def exp_table1(reps=1000, blocks=20, d=0.0, tag=""):
    print(f"TABLE 1  (N=10 miners, {blocks} blocks, {reps} repetitions, network delay {d * 1e3:.0f} ms)", flush=True)
    keys = ["L", "E", "D", "MB", "BB", "W"]

    def cw(cfg):
        def f(seed):
            rec = run(cfg, seed)
            for r in rec:
                r["W"] = r["hashes"]
            return rec
        return f

    protos = [
        ("PoW", lambda s: m.pow_blocks(10, blocks, np.random.default_rng(s), d=d)),
        ("PoS", lambda s: m.pos_blocks(blocks)),
        ("PBFT", lambda s: m.pbft_blocks(blocks)),
        ("CW-PoW", cw(ref_basic(10, blocks, net_delay=d))),
        ("Enhanced CW-PoW (useful work)", cw(ref_enhanced(10, blocks, net_delay=d))),
        ("Enhanced CW-PoW (idle waiting)", cw(ref_enhanced(10, blocks, net_delay=d, wait_mode="idle"))),
        ("Green-PoW", lambda s: m.green_blocks(10, blocks, np.random.default_rng(s), d=d)),
        ("Proof-of-Experience", lambda s: m.poe_blocks(10, blocks, np.random.default_rng(s), d=d)),
    ]
    rows = []
    for name, f in protos:
        rr = []
        for k in range(reps):
            rec = f(6000 + k)
            rr.append({kk: float(np.mean([r[kk] for r in rec])) for kk in keys})
        s = {k: (float(np.mean([r[k] for r in rr])),
                 1.96 * float(np.std([r[k] for r in rr], ddof=1)) / math.sqrt(reps)) for k in keys}
        print(f"  {name:32s} L={s['L'][0] * 1e3:8.1f}+-{s['L'][1] * 1e3:5.1f} ms  Th={1 / s['L'][0]:7.2f}  W={s['W'][0] / 1e3:7.1f}k  "
              f"E={s['E'][0]:7.2f}+-{s['E'][1]:.2f}  D={s['D'][0]:.3f}  MB={s['MB'][0]:6.1f}  BB={s['BB'][0]:10.0f}", flush=True)
        rows.append([name] + [s[k][0] for k in keys] + [s[k][1] for k in keys])
    write_csv(f"des_table1{tag}.csv", ["protocol"] + [f"mean_{k}" for k in keys] + [f"ci95_{k}" for k in keys], rows)


# ------------------------------------------------------------------------------------------------
# mechanism ablation
# ------------------------------------------------------------------------------------------------
def variants(**common):
    """Mechanism variants. `common` overrides (network delay, round length, ...) apply to all of them."""
    full_uw = Cfg(name="Full (useful work)", **common)
    full_dc = replace(full_uw, name="Full (duty-cycling)", wait_mode="sleep")
    full_idle = replace(full_uw, name="Full (idle waiting)", wait_mode="idle")
    b0 = basic_cfg(name="B0: all mechanisms off", R0=6, S0=3, **common)
    v = [
        ("B0", b0),
        ("FULL_UW", full_uw),
        ("FULL_UW_CAL", replace(full_uw, name="Full(UW), APT target 0.25 s", apt_target_latency=0.25)),
        ("FULL_DC", full_dc),
        ("FULL_IDLE", full_idle),
        ("FULL_IDLE63", replace(full_idle, name="Full (idle waiting), fixed (6,3)", use_apt=False)),
        # one mechanism removed from Full(UW)
        ("UW-APT63", replace(full_uw, name="Full(UW) - APT, fixed (6,3)", use_apt=False)),
        ("UW-APT126", replace(full_uw, name="Full(UW) - APT, fixed (12,6)", use_apt=False, R0=12, S0=6)),
        ("UW-APT121", replace(full_uw, name="Full(UW) - APT, fixed (12,1)", use_apt=False, R0=12, S0=1)),
        ("UW-CR", replace(full_uw, name="Full(UW) - commit-reveal", use_cr=False)),
        ("UW-PIPE", replace(full_uw, name="Full(UW) - pipelining", use_pipe=False)),
        ("UW+PIPEH", replace(full_uw, name="Full(UW), precomputation hidden behind hashing", pipe_overlap="hash")),
        ("UW-UW", replace(full_uw, name="Full(UW) - useful work (idle)", wait_mode="idle")),
        ("UW-COM", replace(full_uw, name="Full(UW) - committee", use_committee=False)),
        # one mechanism removed from Full(DC)
        ("DC-APT63", replace(full_dc, name="Full(DC) - APT, fixed (6,3)", use_apt=False)),
        ("DC-APT121", replace(full_dc, name="Full(DC) - APT, fixed (12,1)", use_apt=False, R0=12, S0=1)),
        ("DC-CR", replace(full_dc, name="Full(DC) - commit-reveal", use_cr=False)),
        ("DC-PIPE", replace(full_dc, name="Full(DC) - pipelining", use_pipe=False)),
        ("DC-DC", replace(full_dc, name="Full(DC) - duty-cycling (idle)", wait_mode="idle")),
        ("DC-COM", replace(full_dc, name="Full(DC) - committee", use_committee=False)),
        # one mechanism added to B0
        ("B0+APT", replace(b0, name="B0 + APT", use_apt=True)),
        ("B0+CR", replace(b0, name="B0 + commit-reveal", use_cr=True)),
        ("B0+PIPE", replace(b0, name="B0 + pipelining (wait overlap)", use_pipe=True)),
        ("B0+PIPEH", replace(b0, name="B0 + pipelining (hash overlap)", use_pipe=True, pipe_overlap="hash")),
        ("B0+UW", replace(b0, name="B0 + useful work", wait_mode="uw")),
        ("B0+DC", replace(b0, name="B0 + duty-cycling", wait_mode="sleep")),
        ("B0+COM", replace(b0, name="B0 + committee", use_committee=True)),
    ]
    return v


def exp_ablation(reps=1000, skip=10, tag="", **common):
    label = f" [{tag}]" if tag else ""
    print(f"ABLATION{label} (N=10, {reps} repetitions x 30 blocks, steady state = blocks 11-30)", flush=True)
    rows = []
    for key, cfg in variants(**common):
        res = summarize(replicate(cfg, reps, skip=skip))
        print(f"  {key:10s} L={res['L'][0] * 1e3:6.1f}+-{ci(res, 'L') * 1e3:4.1f} ms  E={res['E'][0]:6.2f}+-{ci(res, 'E'):4.2f} "
              f"[hash {res['E_hash'][0]:5.1f} ovh {res['E_ovh'][0]:4.1f} uw {res['E_uw'][0]:5.1f} waste {res['E_waste'][0]:5.2f}] "
              f"D={res['D'][0]:.3f} MB={res['MB'][0]:4.0f} R={res['R'][0]:4.1f}", flush=True)
        rows.append([key, cfg.name] + [f"{res[k][0]:.6g}" for k in KEYS] + [f"{res[k][2]:.6g}" for k in KEYS])
    # PoW race at the same expected work as the six-round basic structure
    rng = np.random.default_rng(77)
    n = common.get("n", 10)
    d = common.get("net_delay", 0.0)
    t_round = common.get("t_round", 0.020)
    # same expected work as B0: the mean number of hashes of the whole network in one B0 block
    b0cfg = variants(**common)[0][1]
    W = float(np.mean([run(replace(b0cfg, blocks=1), 31000 + k)[0]["hashes"] for k in range(2000)]))
    pw = [m.pow_block(n, W, t_round * P_HIT_ENH, d, rng) for _ in range(20000)]
    Lp, Ep = float(np.mean([x["L"] for x in pw])), float(np.mean([x["E"] for x in pw]))
    sdL = float(np.std([x["L"] for x in pw], ddof=1)) / math.sqrt(len(pw)) * 1.96
    sdE = float(np.std([x["E"] for x in pw], ddof=1)) / math.sqrt(len(pw)) * 1.96
    print(f"  POW_RACE   L={Lp * 1e3:6.1f} ms  E={Ep:6.2f}  (same expected work as B0: {W / 1e3:.1f}k hashes, D=1)", flush=True)
    row = ["POW_RACE", "PoW race, same expected work as B0"] + ["0"] * len(KEYS) + ["0"] * len(KEYS)
    row[2 + KEYS.index("L")] = f"{Lp:.6g}"
    row[2 + KEYS.index("E")] = f"{Ep:.6g}"
    row[2 + KEYS.index("E_hash")] = f"{Ep:.6g}"
    row[2 + KEYS.index("D")] = "1"
    row[2 + len(KEYS) + KEYS.index("L")] = f"{sdL:.6g}"
    row[2 + len(KEYS) + KEYS.index("E")] = f"{sdE:.6g}"
    rows.append(row)
    write_csv(f"des_ablation{tag}.csv", ["variant", "name"] + [f"mean_{k}" for k in KEYS] + [f"ci95_{k}" for k in KEYS], rows)


# ------------------------------------------------------------------------------------------------
# adaptive parameter tuning
# ------------------------------------------------------------------------------------------------
def exp_apt(reps=200):
    print("APT: latency-target sweep (steady state = blocks 21-60), original rule versus priority rule", flush=True)
    rows = []
    for rule in ("original", "priority"):
        for tgt in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.00):
            cfg = replace(Cfg(), blocks=60, apt_target_latency=tgt, apt_rule=rule)
            rr = [run(cfg, 2000 + k) for k in range(reps)]
            Rm = float(np.mean([[b["R"] for b in r[20:]] for r in rr]))
            Rsd = float(np.mean([np.std([b["R"] for b in r[20:]]) for r in rr]))
            Lm = float(np.mean([[b["L"] for b in r[20:]] for r in rr]))
            Em = float(np.mean([[b["E"] for b in r[20:]] for r in rr]))
            print(f"  {rule:9s} target {tgt:4.2f} s : NbrR = {Rm:5.2f} (within-run sd {Rsd:.2f})  L = {Lm * 1e3:6.1f} ms  E = {Em:6.2f}", flush=True)
            rows.append([rule, tgt, Rm, Rsd, Lm, Em])
    write_csv("des_apt_target_sweep.csv", ["rule", "target_latency_s", "NbrR_mean", "NbrR_sd_within", "L_mean_s", "E_mean"], rows)

    print("APT: regime change, +25 ms propagation delay in blocks 21-40 (target 0.25 s)", flush=True)
    delay, win, blocks, tgt = 0.025, (20, 40), 60, 0.25
    sched = lambda b: delay if win[0] <= b < win[1] else 0.0
    rows, summary = [], []
    cases = [("APT, original rule", dict(use_apt=True, apt_rule="original")),
             ("APT, priority rule", dict(use_apt=True, apt_rule="priority")),
             ("static (6,3)", dict(use_apt=False)),
             ("static (12,6)", dict(use_apt=False, R0=12, S0=6))]
    for label, kw in cases:
        cfg = replace(Cfg(), blocks=blocks, apt_target_latency=tgt, delay_schedule=sched, **kw)
        rr = [run(cfg, 3000 + k) for k in range(reps)]
        for b in range(blocks):
            rows.append([label, b + 1, np.mean([r[b]["R"] for r in rr]), np.mean([r[b]["L"] for r in rr]),
                         np.mean([r[b]["E"] for r in rr])])
        inside = range(win[0] + 6, win[1])          # after the controller has reacted
        outside = list(range(6, win[0])) + list(range(win[1] + 6, blocks))
        Lin = float(np.mean([[r[b]["L"] for b in inside] for r in rr]))
        Lout = float(np.mean([[r[b]["L"] for b in outside] for r in rr]))
        Rin = float(np.mean([[r[b]["R"] for b in inside] for r in rr]))
        Rout = float(np.mean([[r[b]["R"] for b in outside] for r in rr]))
        Ein = float(np.mean([[r[b]["E"] for b in inside] for r in rr]))
        Eout = float(np.mean([[r[b]["E"] for b in outside] for r in rr]))
        print(f"  {label:20s} inside: NbrR {Rin:5.2f} L {Lin * 1e3:6.1f} ms E {Ein:6.2f}   outside: NbrR {Rout:5.2f} L {Lout * 1e3:6.1f} ms E {Eout:6.2f}", flush=True)
        summary.append([label, Rin, Lin, Ein, Rout, Lout, Eout])
    write_csv("des_apt_regime.csv", ["case", "block", "NbrR_mean", "L_mean_s", "E_mean"], rows)
    write_csv("des_apt_regime_summary.csv", ["case", "R_inside", "L_inside_s", "E_inside", "R_outside", "L_outside_s", "E_outside"], summary)

    print("APT: saturation from different starting points (original targets, 12 miners)", flush=True)
    rows = []
    for R0, S0 in [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (6, 1), (6, 6), (2, 6)]:
        cfg = replace(Cfg(), n=12, blocks=30, R0=R0, S0=S0)
        rr = [run(cfg, 4000 + k) for k in range(reps)]
        Rt = np.array([[b["R"] for b in r] for r in rr]).mean(axis=0)
        St = np.array([[b["S"] for b in r] for r in rr]).mean(axis=0)
        bR = int(np.argmax(Rt >= 11.999)) + 1 if (Rt >= 11.999).any() else -1
        bS = int(np.argmax(St <= 1.001)) + 1 if (St <= 1.001).any() else -1
        L8 = float(np.mean([r[7]["L"] for r in rr]))
        E8 = float(np.mean([r[7]["E"] for r in rr]))
        print(f"  ({R0:2d},{S0}) : NbrR = 12 first reached at block {bR:2d}, NbrS = 1 at block {bS:2d};  block 8: L = {L8 * 1e3:6.1f} ms  E = {E8:6.2f}", flush=True)
        rows.append([R0, S0, bR, bS, L8, E8, Rt[-1], St[-1]])
    write_csv("des_apt_saturation.csv", ["R0", "S0", "block_R12", "block_S1", "L_block8_s", "E_block8", "R_final", "S_final"], rows)


# ------------------------------------------------------------------------------------------------
# pipelining
# ------------------------------------------------------------------------------------------------
def exp_pipe(reps=1000):
    print("PIPELINING: saving versus header cost t_hdr and network delay d (R=6, S=3, idle waiting, phi=0.2)", flush=True)
    print("  modes: off | wait = precompute during the wait phase | hash = precompute while the previous round is hashed", flush=True)
    rows = []
    base = basic_cfg(blocks=1, R0=6, S0=3, use_cr=True, use_committee=True, name="pipe")
    for d in (0.0, 0.005, 0.020):
        for hdr in (0.0005, 0.001, 0.002, 0.005, 0.010):
            res = {}
            for mode in ("off", "wait", "hash"):
                cfg = replace(base, use_pipe=(mode != "off"), pipe_overlap=("hash" if mode == "hash" else "wait"),
                              hdr_cost=hdr, net_delay=d)
                res[mode] = summarize(replicate(cfg, reps))
            bound = 5 * (1 - 0.2) * hdr
            out = [d, hdr, res["off"]["L"][0], res["off"]["E"][0], res["off"]["wait_mean"][0], bound]
            line = f"  d={d * 1e3:4.1f} ms  t_hdr={hdr * 1e3:5.1f} ms  L(off)={res['off']['L'][0] * 1e3:7.2f} ms  bound={bound * 1e3:5.2f} ms  "
            for mode in ("wait", "hash"):
                dL = res["off"]["L"][0] - res[mode]["L"][0]
                dE = 100 * (res[mode]["E"][0] - res["off"]["E"][0]) / res["off"]["E"][0]
                out += [res[mode]["L"][0], dL, 100 * dL / res["off"]["L"][0], dE]
                line += f"| {mode}: saves {dL * 1e3:5.2f} ms ({100 * dL / res['off']['L'][0]:4.2f}%) dE {dE:+5.2f}% "
            print(line, flush=True)
            rows.append(out)
    write_csv("des_pipelining.csv", ["net_delay_s", "t_hdr_s", "L_off_s", "E_off", "mean_wait_s", "bound_s",
                                     "L_wait_s", "saving_wait_s", "saving_wait_pct", "dE_wait_pct",
                                     "L_hash_s", "saving_hash_s", "saving_hash_pct", "dE_hash_pct"], rows)


# ------------------------------------------------------------------------------------------------
# duty cycling
# ------------------------------------------------------------------------------------------------
def exp_dc(reps=600):
    t_be = WAKE_THRESHOLD + (E_WAKE + P_IDLE * 0.001) / (P_IDLE - P_SLEEP)
    print(f"DUTY CYCLING: break-even wait t_be = theta + (E_wake + P_idle t_wake)/(P_idle - P_sleep) = {t_be * 1e3:.1f} ms", flush=True)
    base = basic_cfg(blocks=1, R0=6, S0=3, use_cr=True, use_committee=True, use_pipe=True, name="dc")
    idle = summarize(replicate(replace(base, wait_mode="idle"), reps))
    rows = []
    print("  wake-up energy sweep (t_round = 20 ms)", flush=True)
    for ew in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3, 1.0):
        dc = summarize(replicate(replace(base, wait_mode="sleep", e_wake=ew), reps))
        d = 100 * (dc["E"][0] - idle["E"][0]) / idle["E"][0]
        dw = 100 * (dc["E_waste"][0] - idle["E_waste"][0]) / idle["E_waste"][0]
        print(f"    E_wake={ew:6.3f}: E idle {idle['E'][0]:.3f}  sleep {dc['E'][0]:.3f} ({d:+6.2f}%)   waste idle {idle['E_waste'][0]:.4f} -> sleep {dc['E_waste'][0]:.4f} ({dw:+7.1f}%)   wakes/block {dc['wakes'][0]:.2f}", flush=True)
        rows.append([ew, idle["E"][0], dc["E"][0], d, idle["E_waste"][0], dc["E_waste"][0], dw, dc["wakes"][0],
                     ci(idle, "E"), ci(dc, "E")])
    write_csv("des_wake_sweep.csv", ["E_wake", "E_idle", "E_sleep", "dE_pct", "waste_idle", "waste_sleep", "dwaste_pct", "wakes_per_block", "ci_idle", "ci_sleep"], rows)

    print("  round-length sweep (E_wake = 1, P_sleep = 0.5): where does duty cycling start to pay?", flush=True)
    rows = []
    for tr in (0.005, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        b = basic_cfg(blocks=1, R0=6, S0=3, use_cr=True, use_committee=True, use_pipe=True, t_round=tr, name="dcr")
        i = summarize(replicate(replace(b, wait_mode="idle"), max(150, reps // 3)))
        s = summarize(replicate(replace(b, wait_mode="sleep"), max(150, reps // 3)))
        u = summarize(replicate(replace(b, wait_mode="uw"), max(150, reps // 3)))
        d = 100 * (s["E"][0] - i["E"][0]) / i["E"][0]
        du = 100 * (u["E"][0] - i["E"][0]) / i["E"][0]
        print(f"    t_round={tr * 1e3:8.1f} ms: mean wait {i['wait_mean'][0] * 1e3:9.2f} ms  E idle {i['E'][0]:9.3f}  sleep {s['E'][0]:9.3f} ({d:+6.2f}%)  uw {u['E'][0]:9.3f} ({du:+6.1f}%)  "
              f"waste idle {i['E_waste'][0]:8.3f} sleep {s['E_waste'][0]:8.3f}", flush=True)
        rows.append([tr, i["wait_mean"][0], i["E"][0], s["E"][0], u["E"][0], d, du, i["E_waste"][0], s["E_waste"][0],
                     i["L"][0], s["L"][0], u["L"][0], i["D"][0]])
    write_csv("des_dc_roundlength.csv", ["t_round_s", "mean_wait_s", "E_idle", "E_sleep", "E_uw", "dE_sleep_pct", "dE_uw_pct", "waste_idle",
                                         "waste_sleep", "L_idle", "L_sleep", "L_uw", "D_idle"], rows)

    print("  idle-energy share: simulation versus eps(D) = (1-D)/(rho*D + 1 - D)", flush=True)
    rho = P_ACTIVE / P_IDLE
    rows = []
    for tr in (0.005, 0.02, 0.1):
        for R0 in (3, 6, 12):
            b = basic_cfg(blocks=1, R0=R0, S0=3, t_round=tr, hdr_cost=0.0, name="share")
            s = summarize(replicate(b, 300))
            Dm = s["D"][0]
            eps_model = (1 - Dm) / (rho * Dm + 1 - Dm)
            eps_sim = s["E_waste"][0] / s["E"][0]
            rows.append([tr, R0, Dm, eps_sim, eps_model])
            print(f"    t_round={tr * 1e3:5.1f} ms R={R0:2d}: D={Dm:.3f}  idle share simulated {eps_sim:.4f}  formula {eps_model:.4f}", flush=True)
    write_csv("des_idle_share.csv", ["t_round_s", "R", "D", "eps_sim", "eps_formula"], rows)


# ------------------------------------------------------------------------------------------------
# scalability
# ------------------------------------------------------------------------------------------------
def exp_scal(reps=30):
    print("SCALABILITY (reference configurations of Table 1, steady state = blocks 11-30; identical per-miner capacity)", flush=True)
    rows = []
    for n in (10, 50, 100, 500, 1000):
        for label, cfg in [("CW-PoW", ref_basic(n, 30)), ("Enhanced CW-PoW", ref_enhanced(n, 30, use_apt=False, wait_mode="idle"))]:   # fixed (6,3): the outcome of APT depends on N through the message count
            rr = summarize(replicate(cfg, reps if n < 500 else max(10, reps // 3), skip=10))
            print(f"  N={n:5d} {label:16s} L={rr['L'][0] * 1e3:7.1f} ms  E={rr['E'][0]:7.2f}  D={rr['D'][0]:.3f}  MB={rr['MB'][0]:9.0f}  BB={rr['BB'][0]:11.0f}  R={rr['R'][0]:.1f} S={rr['S'][0]:.1f}", flush=True)
            rows.append([n, label] + [rr[k][0] for k in KEYS] + [rr[k][2] for k in KEYS])
    write_csv("des_scalability.csv", ["N", "protocol"] + [f"mean_{k}" for k in KEYS] + [f"ci95_{k}" for k in KEYS], rows)


# ------------------------------------------------------------------------------------------------
# reveal withholding, propagation spikes
# ------------------------------------------------------------------------------------------------
def exp_byz(reps=600):
    print("REVEAL WITHHOLDING (12 miners, fixed (6,3), all mechanisms, useful work)", flush=True)
    rows = []
    for tau in (0.005, 0.010, 0.020, 0.050):
        for k in (0, 1, 2, 4, 5):
            cfg = replace(Cfg(), n=12, blocks=1, use_apt=False, n_withhold=k, reveal_timeout=tau)
            rr = summarize(replicate(cfg, reps))
            print(f"  timeout {tau * 1e3:5.1f} ms  withholders {k} of 12: L={rr['L'][0] * 1e3:6.1f} ms  E={rr['E'][0]:6.2f}  MB={rr['MB'][0]:5.0f}", flush=True)
            rows.append([tau, k, rr["L"][0], rr["E"][0], rr["MB"][0]])
    write_csv("des_withholding.csv", ["timeout_s", "withholders", "L_s", "E", "MB"], rows)
    print("PROPAGATION SPIKES (12 miners, fixed (6,3), all mechanisms)", flush=True)
    rows = []
    for p, dur in [(0.0, 0.0), (0.1, 0.05), (0.3, 0.05), (0.3, 0.15)]:
        cfg = replace(Cfg(), n=12, blocks=1, use_apt=False, spike_prob=p, spike_dur=dur)
        rr = summarize(replicate(cfg, reps))
        print(f"  p={p:.1f} duration={dur * 1e3:5.0f} ms: L={rr['L'][0] * 1e3:7.1f} ms  E={rr['E'][0]:6.2f}", flush=True)
        rows.append([p, dur, rr["L"][0], rr["E"][0]])
    write_csv("des_spikes.csv", ["spike_prob", "spike_dur_s", "L_s", "E"], rows)


# ------------------------------------------------------------------------------------------------
# comparison with PoW at equal mean block time, and fairness under heterogeneous hash power
# ------------------------------------------------------------------------------------------------
def protocol_cfg(n, S, R=6, **kw):
    """CW-PoW as specified in Section 4 with the structure of the evaluation: fixed (R, S), commit-reveal,
    pipelining and committee, idle waiting (the miners that stand down wait in the same state)."""
    return replace(basic_cfg(n=n, blocks=1, R0=R, S0=S, **kw), use_cr=True, use_pipe=True, use_committee=True,
                   name="CW-PoW (R=%d, S=%d)" % (R, S))


def jain(F):
    F = np.asarray(F, dtype=float)
    return float(F.sum() ** 2 / (len(F) * (F ** 2).sum()))


def exp_equal(reps=400):
    """PoW is given the same mean block time as CW-PoW (its difficulty is set accordingly).  All PoW miners hash at P_a
    during the whole block time, so E_PoW = P_a * mean(L); its block time is d + Exp(mean(L) - d)."""
    print("EQUAL MEAN BLOCK TIME: CW-PoW (fixed R=6) versus PoW tuned to the same mean block time (identical miners)", flush=True)
    rows = []
    for n in (10, 100, 1000):
        for S in (3, 5, 10):
            for d in (0.0, 0.020):
                cfg = protocol_cfg(n, S, net_delay=d)
                rr = max(100, reps if n < 1000 else reps // 4)
                rec = [run(cfg, 8000 + k)[0] for k in range(rr)]
                L = np.array([r["L"] for r in rec])
                E, MB, BB, Wc = (float(np.mean([r[k] for r in rec])) for k in ("E", "MB", "BB", "hashes"))
                Lm = float(L.mean())
                p99c = float(np.percentile(L, 99))
                p99p = d + (Lm - d) * math.log(100.0)
                Ep = P_ACTIVE * Lm
                Wp = n * Lm / TAU                    # hashes of the whole PoW network in one block
                print(f"  N={n:5d} S={S:2d} d={d * 1e3:3.0f} ms: CW L={Lm * 1e3:7.1f} ms p99={p99c * 1e3:7.1f} E={E:6.2f} M={MB:7.0f} | "
                      f"PoW p99={p99p * 1e3:7.1f} E={Ep:6.2f} | dE={100 * (E / Ep - 1):+6.1f}% dp99={100 * (p99c / p99p - 1):+6.1f}%", flush=True)
                rows.append([n, S, d, Lm, p99c, E, MB, BB, p99p, Ep, 100 * (E / Ep - 1), 100 * (p99c / p99p - 1), Wc, Wp])
    write_csv("des_equal.csv", ["N", "S", "d_s", "L_mean_s", "L_p99_s", "E", "MB", "BB", "pow_p99_s", "pow_E", "dE_pct", "dp99_pct", "W_cw", "W_pow"], rows)


def exp_fairness(blocks_small=40000):
    """Fairness under heterogeneous hash power (log-normal, sigma = 0.5).  F_i = P_i / S_i, with P_i the winning
    frequency and S_i the share of the hash power.  The miners are grouped into deciles by power (quintiles for N = 10)."""
    print("FAIRNESS (heterogeneous power, sigma = 0.5; winner = lowest final hash among the miners of the last round)", flush=True)
    rows = []
    plan = {10: (blocks_small, (2, 3, 5, 10)), 100: (int(blocks_small * 1.5), (3, 10, 25, 50, 100)),
            1000: (blocks_small, (3, 10, 50, 250))}
    for n, (B, Ss) in plan.items():
        sp = np.exp(np.random.default_rng(3).normal(0, 0.5, n))
        share = sp / sp.sum()
        ref = np.random.default_rng(1).multinomial(B, share) / B / share
        groups = np.array_split(np.argsort(sp), 5 if n == 10 else 10)
        rows.append([n, "reference", B, jain(ref), ref.min(), ref.max()] + [float(ref[g].mean()) for g in groups] + [""] * (10 - len(groups)) + ["", ""])
        for S in Ss:
            cfg = replace(protocol_cfg(n, S), speeds=tuple(sp))
            wins = np.zeros(n)
            inc = np.zeros(n)
            Ls, Es = [], []
            for k in range(B):
                r = run(cfg, 9000 + k)[0]
                wins[r["winner_first"]] += 1
                inc[r["part1"]] += 1
                Ls.append(r["L"])
                Es.append(r["E"])
            F = wins / B / share
            Fg = [float(wins[g].sum() / B / share[g].sum()) for g in groups]
            Pg = [float(inc[g].sum() / B / (min(S, n) * share[g].sum())) for g in groups]
            print(f"  N={n:5d} S={S:4d}: L={np.mean(Ls) * 1e3:7.1f} ms E={np.mean(Es):6.2f} | Jain={jain(F):.3f} min={F.min():.2f} max={F.max():.2f} | "
                  f"F by decile: " + " ".join(f"{x:.2f}" for x in Fg), flush=True)
            rows.append([n, S, B, jain(F), float(F.min()), float(F.max())] + Fg + [""] * (10 - len(Fg)) + [float(np.mean(Ls)), float(np.mean(Es))])
            rows.append([n, f"{S}_entry", B, "", "", ""] + Pg + [""] * (10 - len(Pg)) + ["", ""])
            if n == 100 and S in (3, 10):
                cf = replace(cfg, winner_rule="first")
                wf = np.zeros(n)
                for k in range(B):
                    wf[run(cf, 9000 + k)[0]["winner_first"]] += 1
                Ff = wf / B / share
                Fgf = [float(wf[g].sum() / B / share[g].sum()) for g in groups]
                print(f"  N={n:5d} S={S:4d} first-finisher rule: Jain={jain(Ff):.3f} min={Ff.min():.2f} max={Ff.max():.2f} | F by decile: " + " ".join(f"{x:.2f}" for x in Fgf), flush=True)
                rows.append([n, f"{S}_first", B, jain(Ff), float(Ff.min()), float(Ff.max())] + Fgf + [""] * (10 - len(Fgf)) + ["", ""])
    write_csv("des_fairness.csv", ["N", "S", "blocks", "jain", "F_min", "F_max"] + [f"g{i}" for i in range(1, 11)] + ["L_mean_s", "E_mean"], rows)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    if which in ("all", "validate"):
        exp_validate()
    if which in ("all", "table1"):
        exp_table1(reps)
        exp_table1(reps, d=0.010, tag="_d10ms")
    if which in ("all", "ablation"):
        exp_ablation(reps)
        exp_ablation(max(300, reps // 3), tag="_d2ms", net_delay=0.002)
        exp_ablation(max(300, reps // 3), tag="_d10ms", net_delay=0.010)
        exp_ablation(max(300, reps // 3), tag="_t100ms", t_round=0.100)
    if which in ("all", "apt"):
        exp_apt(max(100, reps // 5))
    if which in ("all", "pipe"):
        exp_pipe(reps)
    if which in ("all", "dc"):
        exp_dc(max(300, reps // 2))
    if which in ("all", "scal"):
        exp_scal(max(10, reps // 30))
    if which in ("all", "byz"):
        exp_byz(max(300, reps // 2))
    if which in ("all", "equal"):
        exp_equal(max(200, reps // 2))
    if which in ("all", "fair"):
        exp_fairness()
    print(f"done in {time.time() - t0:.0f} s")
