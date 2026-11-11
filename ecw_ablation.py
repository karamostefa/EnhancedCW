"""
ecw_ablation.py
===============
Mechanism-level ablation of Enhanced CW-PoW with the THREADED prototype (cross-check of manuscript Section 5.10,
Table A2; the reference results come from the event-driven model ecw_des.py).

The five mechanisms of the enhanced protocol are individually switchable here:

    APT        adaptive tuning of (NbrR, NbrS) after every block
    C-R        commit-reveal round identifier (+ reveal message, +1 message flight per round)
    PIPE       partial pipelining: header pre-computation hidden in the wait phase
    WAIT       idle-time policy: 'useful_work' | 'sleep' (duty-cycling) | 'idle' (awake, nothing)
    COMMITTEE  hybrid committee finalization (verification + votes)

The module re-uses the constants and the threaded round structure of the paper's
own simulator ("EnhancedCW 5 algo.py"), so that every variant is measured in the
same environment that produced Table 1. Differences with respect to the legacy
function run_enhanced_cw_pow():

  * every mechanism has an explicit switch (the legacy code only had C-R / useful work as globals);
  * energy is accounted per power state with explicit units:
        E_i = P_active*(t_hash + t_header + t_precompute + t_usefulwork + t_verify)
            + P_idle*t_idle + P_sleep*t_sleep + E_wake*n_wake
    (the legacy accounting bills sleeping time at P_idle, never uses P_sleep, and counts a
    useful-work interval both as compute and as wait); a 'legacy' accounting mode is kept for
    validation against run_enhanced_cw_pow();
  * pipelining is modelled (the legacy code only noted it): after IDRound is known a miner must
    assemble the block header (cost HDR_COST); with pipelining the IDRound-independent part
    (1-HDR_FIN_FRAC of it) is computed during the wait, so only HDR_FIN_FRAC*HDR_COST remains
    on the critical path;
  * block latency L is measured at block level (wall clock from block start to finalization),
    and the frequency of near-simultaneous candidate blocks (fork opportunities) is recorded.

All costs that the legacy simulator does not model are explicit, documented parameters (see Cfg).
Absolute values depend on the host (wall-clock threads); the analysis concerns relative effects.

Run:  python ecw_ablation.py [reps]      (default reps = 20)
"""
import importlib.util
import os
import sys
import csv
import math
import time
import random
import hashlib
import threading
import statistics
from dataclasses import dataclass, replace, field
from math import ceil

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ecw_main_sim", os.path.join(HERE, "EnhancedCW 5 algo.py"))
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

ENH_TARGET = int("25000000000000000000000000000000000000000000000000000000000000", 16)
now = time.perf_counter


@dataclass
class Cfg:
    name: str = "Full (useful work)"
    num_miners: int = 10
    blocks: int = 5
    nbrR0: int = sim.INITIAL_NBRR
    nbrS0: int = sim.INITIAL_NBRS
    use_apt: bool = True
    use_commit_reveal: bool = True
    use_pipelining: bool = True
    wait_mode: str = "useful_work"          # 'useful_work' | 'sleep' | 'idle'
    use_committee: bool = True
    committee_size: int = sim.COMMITTEE_SIZE
    target: int = ENH_TARGET
    # ---- power model (units/s, units) : identical to the Appendix of the manuscript
    p_active: float = sim.P_ACTIVE
    p_idle: float = sim.P_IDLE
    p_sleep: float = sim.P_SLEEP
    e_wake: float = sim.E_WAKE
    wake_threshold: float = sim.WAKE_THRESHOLD   # idle this long before going to sleep
    t_wake: float = 0.001                         # wake-up latency (s), duty-cycling only
    uw_unit: float = sim.USEFUL_WORK_COST         # one useful-work unit (s, active)
    uw_verify: float = sim.USEFUL_WORK_VERIFY     # verification of useful work per round (s, active)
    # ---- costs not modelled by the legacy simulator (explicit parameters)
    hdr_cost: float = 0.002                       # full header assembly + mid-state per round (s)
    hdr_fin_frac: float = 0.2                     # share of hdr_cost that depends on IDRound
    reveal_delay: float = 0.001                   # extra message flight per round when C-R is used (s)
    verify_cost: float = 0.001                    # committee member verification (s, active)
    vote_delay: float = 0.001                     # vote propagation (s)
    # ---- APT controller (same rules as sim.adaptive_tuner, targets are arguments here)
    apt_target_latency: float = sim.TARGET_LATENCY_PER_BLOCK
    apt_target_energy: float = sim.TARGET_ENERGY_PER_ROUND
    apt_target_msgs: int = sim.TARGET_MSGS_PER_BLOCK
    apt_bounds: tuple = (sim.APT_NBRR_MIN, sim.APT_NBRR_MAX, sim.APT_NBRS_MIN, sim.APT_NBRS_MAX)
    # ---- disturbance: extra per-round propagation delay (s) as a function of the block index
    delay_schedule: object = None
    accounting: str = "explicit"                  # 'explicit' | 'legacy'
    fork_windows: tuple = (0.002, 0.005, 0.010, 0.020)


def _apt(cfg, R, S, obs):
    r_min, r_max, s_min, s_max = cfg.apt_bounds
    if obs["lat"] > cfg.apt_target_latency:
        R = max(r_min, R - 1)
    elif obs["lat"] < 0.5 * cfg.apt_target_latency:
        R = min(r_max, R + 1)
    if obs["e_round"] > cfg.apt_target_energy:
        R = min(r_max, R + 1)
    if obs["msgs"] > cfg.apt_target_msgs:
        S = min(s_max, S + 1)
    elif obs["msgs"] < 0.5 * cfg.apt_target_msgs:
        S = max(s_min, S - 1)
    return max(r_min, min(r_max, int(R))), max(s_min, min(s_max, int(S)))


def run_config(cfg: Cfg, seed: int) -> dict:
    n = cfg.num_miners
    rng = random.Random(seed)
    R, S = cfg.nbrR0, cfg.nbrS0
    lock = threading.Lock()
    per_block = []
    wins = [0] * n
    hashes_tot = [0] * n
    traj = []

    for blk in range(cfg.blocks):
        queues = [[] for _ in range(R)]
        commits = [rng.getrandbits(128) for _ in range(n)] if cfg.use_commit_reveal else [None] * n
        committee = rng.sample(range(n), min(cfg.committee_size, n)) if cfg.use_committee else []
        extra_delay = cfg.delay_schedule(blk) if cfg.delay_schedule else 0.0
        keys = ("hash", "hdr", "pre", "uw", "uwver", "ver", "idle", "sleep", "wait", "wake_lat")
        acc = [{k: 0.0 for k in keys} for _ in range(n)]
        cnt = [{"wake": 0, "msgs": 0, "bytes": 0, "uwu": 0, "hashes": 0} for _ in range(n)]
        final_hash = [None] * n
        final_time = [None] * n

        def miner(i):
            a, c = acc[i], cnt[i]
            tgt = cfg.target
            mrng = random.Random(seed * 7919 + blk * 1000 + i * 17)
            x = mrng.randint(10_000, 10 ** 18)
            nonce, round_id = 1, 0
            if cfg.use_commit_reveal:
                c["msgs"] += 1
                c["bytes"] += sim.COMMIT_MSG_BYTES
            for r in range(R):
                # --- header finalisation once IDRound is known (rounds after the first)
                if r > 0 and cfg.hdr_cost > 0:
                    cost = cfg.hdr_cost * (cfg.hdr_fin_frac if cfg.use_pipelining else 1.0)
                    t0 = now(); time.sleep(cost); a["hdr"] += now() - t0
                # --- hashing (hot loop kept as lean as in the legacy simulator)
                t0 = now()
                h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                nh = 1
                stood_down = False
                while h > tgt:
                    if len(queues[r]) >= S:
                        stood_down = True          # the round is closed: S solutions exist and this miner has none
                        break
                    nonce += 1
                    h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                    nh += 1
                a["hash"] += now() - t0
                c["hashes"] += nh
                if stood_down:
                    break                          # stands down until the block is final
                # --- submit (optionally delayed by the network disturbance)
                if extra_delay > 0:
                    t0 = now(); time.sleep(extra_delay); a["wait"] += now() - t0; a["idle"] += now() - t0
                with lock:
                    accepted = len(queues[r]) < S
                    if accepted:
                        queues[r].append((i, nonce))
                if not accepted:
                    break
                c["msgs"] += 1
                c["bytes"] += sim.CW_SOLUTION_MSG_BYTES
                if r == R - 1:
                    final_hash[i] = h
                    final_time[i] = now()
                    c["msgs"] += 1
                    c["bytes"] += sim.CW_BLOCK_MSG_BYTES
                    break
                # --- wait phase
                w0 = now()
                if cfg.use_pipelining and cfg.hdr_cost > 0:
                    t0 = now(); time.sleep(cfg.hdr_cost * (1.0 - cfg.hdr_fin_frac)); a["pre"] += now() - t0
                polls_idle_start = now()
                slept = False
                while True:
                    with lock:
                        quorum = len(queues[r]) >= S
                    if quorum:
                        break
                    if cfg.wait_mode == "useful_work":
                        t0 = now(); time.sleep(cfg.uw_unit); a["uw"] += now() - t0; c["uwu"] += 1
                    elif cfg.wait_mode == "sleep":
                        t0 = now(); time.sleep(sim.SLEEP_POLL); dt = now() - t0
                        if (now() - polls_idle_start) <= cfg.wake_threshold and not slept:
                            a["idle"] += dt
                        else:
                            if not slept:
                                slept = True
                                c["wake"] += 1
                            a["sleep"] += dt
                    else:  # idle
                        t0 = now(); time.sleep(sim.SLEEP_POLL); a["idle"] += now() - t0
                if cfg.use_commit_reveal:
                    t0 = now(); time.sleep(cfg.reveal_delay); dt = now() - t0
                    if cfg.wait_mode == "sleep" and slept:
                        a["sleep"] += dt
                    else:
                        a["idle"] += dt if cfg.wait_mode != "useful_work" else 0.0
                        if cfg.wait_mode == "useful_work":
                            a["uw"] += dt          # miner keeps working while the reveals travel
                    c["msgs"] += 1
                    c["bytes"] += sim.REVEAL_MSG_BYTES
                if cfg.wait_mode == "sleep" and slept and cfg.t_wake > 0:
                    t0 = now(); time.sleep(cfg.t_wake); a["wake_lat"] += now() - t0
                a["wait"] += now() - w0
                if cfg.wait_mode == "useful_work":
                    c["msgs"] += 1
                    c["bytes"] += int(cfg.uw_verify * 1000)
                    if cfg.uw_verify > 0 and cfg.accounting == "explicit":
                        t0 = now(); time.sleep(cfg.uw_verify); a["uwver"] += now() - t0
                with lock:
                    take = min(S, len(queues[r]))
                    s = sum(queues[r][k][1] for k in range(take)) if take > 0 else 0
                if cfg.use_commit_reveal:
                    agg = 0
                    for j in range(n):
                        agg ^= commits[j]
                    round_id = (agg + s) & ((1 << 256) - 1)
                else:
                    round_id = s
                c["msgs"] += 1
                c["bytes"] += sim.CW_ROUNDID_MSG_BYTES
                nonce = 1

        t_block0 = now()
        threads = [threading.Thread(target=miner, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        winner = min((i for i in range(n) if final_hash[i] is not None), key=lambda i: final_hash[i])
        wins[winner] += 1
        if cfg.use_committee and committee:
            time.sleep(cfg.verify_cost + cfg.vote_delay)
            for m in committee:
                acc[m]["ver"] += cfg.verify_cost
                cnt[m]["msgs"] += 1
                cnt[m]["bytes"] += 100
        L = now() - t_block0
        # a miner is powered during the whole block interval: the time after its last action is spent in the
        # wait state of the active mode (same rule as the event-driven model)
        for i in range(n):
            a = acc[i]
            el = sum(a[k] for k in ("hash", "hdr", "pre", "uw", "uwver", "ver", "idle", "sleep", "wake_lat"))
            tail = max(L - el, 0.0)
            if cfg.wait_mode == "useful_work":
                a["uw"] += tail
            elif cfg.wait_mode == "sleep":
                th = min(tail, cfg.wake_threshold)
                a["idle"] += th
                a["sleep"] += tail - th
                if tail > cfg.wake_threshold:
                    cnt[i]["wake"] += 1
            else:
                a["idle"] += tail

        # ---- conflict (fork-opportunity) statistics from final-candidate timestamps
        ft = sorted(t for t in final_time if t is not None)
        gap = (ft[1] - ft[0]) if len(ft) > 1 else float("inf")
        conflicts = {w: (gap < w) for w in cfg.fork_windows}

        # ---- per-miner energy and duty cycle for this block
        E_list, D_list, waste_list, hashE_list, ovhE_list, uwE_list, wasteE_list = [], [], [], [], [], [], []
        for i in range(n):
            a, c = acc[i], cnt[i]
            if cfg.accounting == "legacy":
                comp = a["hash"] + a["uw"] + a["ver"]
                E = comp * cfg.p_active + a["wait"] * cfg.p_idle + cfg.e_wake
                active = comp
                nonact = a["wait"]
                waste = a["wait"] * cfg.p_idle + cfg.e_wake
                ovh = a["ver"] * cfg.p_active
                uwE = a["uw"] * cfg.p_active
            else:
                active = a["hash"] + a["hdr"] + a["pre"] + a["uw"] + a["uwver"] + a["ver"]
                E_act = active * cfg.p_active
                E_waste = a["idle"] * cfg.p_idle + a["sleep"] * cfg.p_sleep + cfg.e_wake * c["wake"] + \
                          a["wake_lat"] * cfg.p_idle
                E = E_act + E_waste
                nonact = a["idle"] + a["sleep"] + a["wake_lat"]
                waste = E_waste
                ovh = (a["hdr"] + a["pre"] + a["uwver"] + a["ver"]) * cfg.p_active
                uwE = a["uw"] * cfg.p_active
            T = active + nonact
            D_list.append(active / T if T > 0 else 1.0)
            E_list.append(E)
            waste_list.append(waste / E if E > 0 else 0.0)
            hashE_list.append(a["hash"] * cfg.p_active)
            ovhE_list.append(ovh)
            uwE_list.append(uwE)
            wasteE_list.append(waste)
            hashes_tot[i] += c["hashes"]

        total_msgs = sum(c["msgs"] for c in cnt)
        # legacy-style observation for the APT controller (kept identical to the paper's controller)
        leg_comp = [acc[i]["hash"] + acc[i]["uw"] + acc[i]["ver"] for i in range(n)]
        leg_wait = [acc[i]["wait"] for i in range(n)]
        obs_lat = sum(leg_comp[i] + leg_wait[i] for i in range(n)) / n
        obs_e_round = (sum(leg_comp[i] * cfg.p_active + leg_wait[i] * cfg.p_idle for i in range(n)) / n) / max(1, R)
        per_block.append({
            "L": L, "E": statistics.mean(E_list), "D": statistics.mean(D_list),
            "waste_frac": statistics.mean(waste_list), "hash_E": statistics.mean(hashE_list),
            "ovh_E": statistics.mean(ovhE_list), "uw_E": statistics.mean(uwE_list),
            "waste_E": statistics.mean(wasteE_list),
            "msgs": total_msgs, "uwu": statistics.mean(c["uwu"] for c in cnt), "R": R, "S": S,
            "conf": conflicts,
        })
        traj.append((R, S, L))
        if cfg.use_apt:
            newR, newS = _apt(cfg, R, S, {"lat": obs_lat, "e_round": obs_e_round, "msgs": total_msgs})
            R = R + (1 if newR > R else -1) if abs(newR - R) > 1 else newR
            S = S + (1 if newS > S else -1) if abs(newS - S) > 1 else newS

    tot_h = sum(hashes_tot) or 1
    F_list = [(wins[i] / cfg.blocks) / (hashes_tot[i] / tot_h) for i in range(n) if hashes_tot[i] > 0]
    out = {
        "L": statistics.mean(b["L"] for b in per_block),
        "E_block": statistics.mean(b["E"] for b in per_block),
        "D": statistics.mean(b["D"] for b in per_block),
        "waste_frac": statistics.mean(b["waste_frac"] for b in per_block),
        "hash_E": statistics.mean(b["hash_E"] for b in per_block),
        "ovh_E": statistics.mean(b["ovh_E"] for b in per_block),
        "uw_E": statistics.mean(b["uw_E"] for b in per_block),
        "waste_E": statistics.mean(b["waste_E"] for b in per_block),
        "MB": statistics.mean(b["msgs"] for b in per_block),
        "UW_units": statistics.mean(b["uwu"] for b in per_block),
        "R_mean": statistics.mean(b["R"] for b in per_block),
        "S_mean": statistics.mean(b["S"] for b in per_block),
        "R_final": R, "S_final": S,
        "F": statistics.mean(F_list) if F_list else float("nan"),
        "traj": traj,
    }
    for w in cfg.fork_windows:
        out[f"conf_{int(w * 1000)}ms"] = statistics.mean(1.0 if b["conf"][w] else 0.0 for b in per_block)
    return out


# ----------------------------------------------------------------------------------------------
# helpers for experiments
# ----------------------------------------------------------------------------------------------
def t_crit(n):
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
             11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
             19: 2.101, 20: 2.093, 25: 2.064, 30: 2.045}
    if n in table:
        return table[n]
    return 1.96 if n > 30 else table[min(table, key=lambda k: abs(k - n))]


def summarize(rows, keys):
    out = {}
    n = len(rows)
    for k in keys:
        vals = [r[k] for r in rows]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if n > 1 else 0.0
        out[k] = (m, sd, t_crit(n) * sd / math.sqrt(n) if n > 1 else 0.0)
    return out


KEYS = ["L", "E_block", "hash_E", "ovh_E", "uw_E", "waste_E", "D", "waste_frac", "MB", "UW_units",
        "R_mean", "S_mean", "F"]


def repeat(cfg, reps, seed0=9000):
    rows = []
    for k in range(reps):
        rows.append(run_config(cfg, seed0 + k))
    return rows


def write_rows(path, header, rows):
    with open(os.path.join(HERE, path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------
# Experiment 0: validation against the legacy simulator (same configuration, legacy accounting)
# ----------------------------------------------------------------------------------------------
def exp_validation(reps=10):
    import io, contextlib
    legacy = []
    for k in range(reps):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = sim.run_enhanced_cw_pow(num_miners=10, runs=5, seed=9000 + k)
        legacy.append({"E_block": r["avg_energy_per_round"] * r["final_params"][0],
                       "TT": r["avg_total_time"], "MB": r["msgs_per_block"]})
    cfg = Cfg(name="re-implementation (legacy accounting)", accounting="legacy", hdr_cost=0.0, reveal_delay=0.0,
              verify_cost=0.001, vote_delay=0.0, uw_verify=0.0)
    mine = []
    for k in range(reps):
        r = run_config(cfg, 9000 + k)
        mine.append({"E_block": r["E_block"], "TT": r["L"], "MB": r["MB"]})
    def ms(rows, key):
        v = [x[key] for x in rows]
        return statistics.mean(v), statistics.stdev(v)
    print("VALIDATION (legacy run_enhanced_cw_pow  vs  ablation re-implementation, legacy accounting)")
    for key in ("E_block", "MB"):
        a, b = ms(legacy, key), ms(mine, key)
        print(f"  {key:8s} legacy {a[0]:8.2f} +- {a[1]:.2f}   re-impl {b[0]:8.2f} +- {b[1]:.2f}   ratio {b[0]/a[0]:.3f}")
    a = ms(legacy, "TT")
    print(f"  per-miner total time per 5 blocks (legacy TT) {a[0]:.3f} +- {a[1]:.3f}")
    return legacy, mine


def exp_validation_basic(reps=10):
    """B0 (all mechanisms off) with the basic CW-PoW target and 5 rounds versus the legacy run_cw_pow()."""
    import io, contextlib
    legacy = []
    for k in range(reps):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = sim.run_cw_pow(num_miners=10, last_round=5, sol_nb=3, runs=5, seed=9000 + k)
        legacy.append({"E_block": r["avg_energy"] * 5, "MB": r["msgs_per_block"]})
    basic_target = int("9000000000000000000000000000000000000000000000000000000000000", 16)
    cfg = Cfg(name="B0 at the basic CW-PoW target", accounting="legacy", use_apt=False, use_commit_reveal=False,
              use_pipelining=False, wait_mode="idle", use_committee=False, nbrR0=5, nbrS0=3, target=basic_target,
              hdr_cost=0.0, reveal_delay=0.0, e_wake=0.0)
    mine = []
    for k in range(reps):
        r = run_config(cfg, 9000 + k)
        mine.append({"E_block": r["E_block"], "MB": r["MB"]})
    def ms(rows, key):
        v = [x[key] for x in rows]
        return statistics.mean(v), statistics.stdev(v)
    print("VALIDATION 2 (legacy run_cw_pow  vs  B0 at the basic target, legacy accounting)")
    for key in ("E_block", "MB"):
        a, b = ms(legacy, key), ms(mine, key)
        print(f"  {key:8s} legacy {a[0]:8.2f} +- {a[1]:.2f}   B0 {b[0]:8.2f} +- {b[1]:.2f}   ratio {b[0]/a[0]:.3f}")
    return legacy, mine


# ----------------------------------------------------------------------------------------------
# Experiment A: mechanism ablation at a common per-round target
# ----------------------------------------------------------------------------------------------
def variants():
    full_uw = Cfg(name="Full (useful work)")
    full_dc = replace(full_uw, name="Full (duty-cycling)", wait_mode="sleep")
    v = [
        ("B0", replace(full_uw, name="B0: all mechanisms off (basic structure)", use_apt=False, use_commit_reveal=False,
                       use_pipelining=False, wait_mode="idle", use_committee=False)),
        ("FULL_UW", full_uw),
        ("FULL_DC", full_dc),
        ("-APT(6,3)", replace(full_uw, name="Full(UW) - APT [fixed (6,3)]", use_apt=False)),
        ("-APT(12,6)", replace(full_uw, name="Full(UW) - APT [fixed (12,6)]", use_apt=False, nbrR0=12, nbrS0=6)),
        ("-CR", replace(full_uw, name="Full(UW) - commit-reveal", use_commit_reveal=False)),
        ("-PIPE", replace(full_uw, name="Full(UW) - pipelining", use_pipelining=False)),
        ("-UW(idle)", replace(full_uw, name="Full(UW) - useful work [idle]", wait_mode="idle")),
        ("-COMMITTEE", replace(full_uw, name="Full(UW) - committee finalization", use_committee=False)),
        ("DC-DC(idle)", replace(full_dc, name="Full(DC) - duty-cycling [idle]", wait_mode="idle")),
        ("FULL_UW_T0.3", replace(full_uw, name="Full(UW), APT target 0.3 s", apt_target_latency=0.3)),
    ]
    # additive view: the baseline B0 plus one mechanism at a time
    b0 = v[0][1]
    v += [
        ("B0+APT", replace(b0, name="B0 + APT", use_apt=True)),
        ("B0+CR", replace(b0, name="B0 + commit-reveal", use_commit_reveal=True)),
        ("B0+PIPE", replace(b0, name="B0 + pipelining", use_pipelining=True)),
        ("B0+UW", replace(b0, name="B0 + useful work", wait_mode="useful_work")),
        ("B0+DC", replace(b0, name="B0 + duty-cycling", wait_mode="sleep")),
        ("B0+COMMITTEE", replace(b0, name="B0 + committee finalization", use_committee=True)),
    ]
    return v


def exp_ablation(reps=20):
    header = ["variant", "name"] + KEYS + ["R_final", "S_final"]
    rows_raw, rows_sum = [], []
    for key, cfg in variants():
        t0 = time.time()
        runs = repeat(cfg, reps)
        for k, r in enumerate(runs):
            rows_raw.append([key, cfg.name, k] + [r[x] for x in KEYS] + [r["R_final"], r["S_final"]])
        s = summarize(runs, KEYS)
        fin = statistics.mean(r["R_final"] for r in runs)
        rows_sum.append([key, cfg.name] + [f"{s[k][0]:.5f}" for k in KEYS] + [f"{fin:.2f}"] +
                        [f"{s[k][2]:.5f}" for k in KEYS])
        print(f"{key:14s} L={s['L'][0]:.3f}+-{s['L'][2]:.3f}  E={s['E_block'][0]:.2f}+-{s['E_block'][2]:.2f} "
              f"[hash {s['hash_E'][0]:.1f} ovh {s['ovh_E'][0]:.1f} uw {s['uw_E'][0]:.1f} waste {s['waste_E'][0]:.1f}]  "
              f"D={s['D'][0]:.3f}  MB={s['MB'][0]:.0f}  UWu={s['UW_units'][0]:.1f} "
              f"R={s['R_mean'][0]:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    write_rows("ablation_mechanisms_raw.csv", ["variant", "name", "rep"] + KEYS + ["R_final", "S_final"], rows_raw)
    write_rows("ablation_mechanisms_summary.csv",
               ["variant", "name"] + [f"mean_{k}" for k in KEYS] + ["R_final_mean"] + [f"ci95_{k}" for k in KEYS],
               rows_sum)


# ----------------------------------------------------------------------------------------------
# Experiment B: APT regulation (targets sweep and regime change)
# ----------------------------------------------------------------------------------------------
def exp_apt_targets(reps=5, blocks=30):
    rows = []
    for tgt in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        cfg = replace(Cfg(name=f"APT target {tgt}"), blocks=blocks, apt_target_latency=tgt)
        runs = repeat(cfg, reps)
        # per-block trajectories, averaged over repetitions
        for b in range(blocks):
            rows.append([tgt, b + 1, statistics.mean(r["traj"][b][0] for r in runs),
                         statistics.mean(r["traj"][b][2] for r in runs)])
        fin = statistics.mean(r["traj"][-1][0] for r in runs)
        lat = statistics.mean(r["L"] for r in runs[:])
        print(f"APT target {tgt:4.1f} s : mean NbrR in last 10 blocks = "
              f"{statistics.mean(r['traj'][b][0] for r in runs for b in range(blocks-10, blocks)):.2f}   mean L = {lat:.3f}",
              flush=True)
    write_rows("apt_target_sweep.csv", ["target_latency_s", "block", "NbrR_mean", "L_mean"], rows)


def exp_apt_regime(reps=5, blocks=36, delay=0.040, window=(12, 24), target=0.3):
    sched = lambda b: delay if window[0] <= b < window[1] else 0.0
    rows = []
    cases = [("APT", dict(use_apt=True)),
             ("static (6,3)", dict(use_apt=False)),
             ("static (12,6)", dict(use_apt=False, nbrR0=12, nbrS0=6))]
    for label, kw in cases:
        cfg = replace(Cfg(name=label), blocks=blocks, apt_target_latency=target, delay_schedule=sched, **kw)
        runs = repeat(cfg, reps)
        for b in range(blocks):
            rows.append([label, b + 1, statistics.mean(r["traj"][b][0] for r in runs),
                         statistics.mean(r["traj"][b][2] for r in runs)])
        L_in = statistics.mean(r["traj"][b][2] for r in runs for b in range(window[0], window[1]))
        L_out = statistics.mean(r["traj"][b][2] for r in runs for b in list(range(0, window[0])) + list(range(window[1], blocks)))
        print(f"{label:14s} mean L during delay regime {L_in:.3f}s, outside {L_out:.3f}s", flush=True)
    write_rows("apt_regime_change.csv", ["case", "block", "NbrR_mean", "L_mean"], rows)


# ----------------------------------------------------------------------------------------------
# Experiment C: pipelining latency saving versus header cost
# ----------------------------------------------------------------------------------------------
def exp_pipelining(reps=15):
    rows = []
    for hdr in (0.0, 0.001, 0.002, 0.005, 0.010):
        res = {}
        for pipe in (False, True):
            cfg = replace(Cfg(name="pipe"), use_apt=False, use_pipelining=pipe, hdr_cost=hdr)
            runs = repeat(cfg, reps)
            res[pipe] = summarize(runs, ["L", "E_block"])
        dL = 100.0 * (res[True]["L"][0] - res[False]["L"][0]) / res[False]["L"][0]
        dE = 100.0 * (res[True]["E_block"][0] - res[False]["E_block"][0]) / res[False]["E_block"][0]
        rows.append([hdr, res[False]["L"][0], res[True]["L"][0], dL, res[False]["E_block"][0], res[True]["E_block"][0], dE])
        print(f"hdr={hdr*1000:4.1f} ms : L {res[False]['L'][0]:.4f} -> {res[True]['L'][0]:.4f}  ({dL:+.1f}%)   "
              f"E {res[False]['E_block'][0]:.2f} -> {res[True]['E_block'][0]:.2f} ({dE:+.1f}%)", flush=True)
    write_rows("pipelining_sweep.csv", ["hdr_cost_s", "L_nopipe", "L_pipe", "dL_pct", "E_nopipe", "E_pipe", "dE_pct"], rows)


# ----------------------------------------------------------------------------------------------
# Experiment D: duty-cycling versus wake-up cost
# ----------------------------------------------------------------------------------------------
def exp_wake(reps=15):
    rows = []
    base = replace(Cfg(name="dc"), use_apt=False)
    idle = summarize(repeat(replace(base, wait_mode="idle"), reps), ["E_block", "L", "waste_frac"])
    for ew in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
        dc = summarize(repeat(replace(base, wait_mode="sleep", e_wake=ew), reps), ["E_block", "L", "waste_frac"])
        d = 100.0 * (dc["E_block"][0] - idle["E_block"][0]) / idle["E_block"][0]
        rows.append([ew, idle["E_block"][0], dc["E_block"][0], d, idle["waste_frac"][0], dc["waste_frac"][0]])
        print(f"E_wake={ew:5.2f} : idle {idle['E_block'][0]:.2f}  sleep {dc['E_block'][0]:.2f} ({d:+.1f}%)  "
              f"waste frac idle {idle['waste_frac'][0]:.3f} -> sleep {dc['waste_frac'][0]:.3f}", flush=True)
    write_rows("wake_sweep.csv", ["E_wake", "E_idle", "E_sleep", "dE_pct", "waste_idle", "waste_sleep"], rows)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "validate"):
        exp_validation()
        exp_validation_basic()
    if which in ("all", "ablation"):
        exp_ablation(reps)
    if which in ("all", "apt"):
        exp_apt_targets()
        exp_apt_regime()
    if which in ("all", "pipe"):
        exp_pipelining()
    if which in ("all", "wake"):
        exp_wake()
