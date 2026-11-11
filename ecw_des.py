"""
ecw_des.py
==========
Event-driven (discrete-event) model of the CW-PoW family.  It is the reference tool for the quantitative results of
the manuscript (Sections 5.5-5.10).  The experiments are in ecw_des_experiments.py:

  * comparison of the protocols (Table 2) and scalability from 10 to 1000 miners;
  * mechanism ablation of Enhanced CW-PoW (each mechanism switched off one at a time, and added one at a time);
  * adaptive parameter tuning (target sweep, regime change, saturation);
  * pipelining (latency saving versus header cost, wait overlap versus hash overlap);
  * duty-cycling (break-even versus wake-up energy and round length);
  * reveal withholding and propagation-delay spikes.

Why a second tool?  "EnhancedCW 5 algo.py" runs every miner as a Python thread and measures wall-clock time.
Its results depend on the host (interpreter-lock contention, timer resolution, background load).  Here every
duration is a model parameter, nothing is measured on the host, and a run is fully determined by its seed.
All constants that exist in the threaded prototype are imported from it (so that the two tools cannot drift
apart); the new parameters are documented in Cfg.

Model (one block = R rounds, quorum S, N identical miners, all start at t = 0)
------------------------------------------------------------------------------
Round r, miner i:
  1. header finalisation (r > 1): duration t_hdr, or phi*t_hdr when pipelining is on          [active]
  2. hashing: X = tau * G, G ~ Geometric(p_hit) attempts until a valid nonce is found          [active]
  3. submission at s_i; the solution is visible to the others after the network delay d
  4. r < R: wait phase.  With pipelining the IDRound-independent part of the next header,
     (1-phi)*t_hdr, is computed either in the wait phase (pipe_overlap = 'wait', active time that overlaps the
     wait) or hidden behind the hashing of the previous round (pipe_overlap = 'hash').  Then the miner waits
     until it sees S solutions of round r (quorum time q_i):
        'idle'  : awake, event driven                                                          [P_idle]
        'sleep' : idle for the threshold theta, then asleep; a wake-up costs E_wake and t_wake  [P_sleep]
        'uw'    : useful-work units of u seconds, checked between units (granularity u)          [active]
  5. commit-reveal: one extra reveal flight d_rev (timeout tau_rev if some reveal is withheld)
  6. useful-work verification (uw mode): t_uwver                                                [active]
r = R: the candidate block is broadcast; no wait.
Block latency L = time at which the last candidate is visible (+ committee verification and vote).
The device of every miner is powered during [0, L]; after its last submission a miner stays in the
wait state of the active mode until the block is finalised (tail).  Hence, for every miner,
E_i = P_a*T_c + P_idle*T_idle + P_sleep*T_sleep + E_wake*n_wake  and  T_c + T_idle + T_sleep = L.

This module only contains the model.  Run the experiments with:  python ecw_des_experiments.py [reps] [which]
"""
import importlib.util
import os
import sys
import csv
import math
import time
from dataclasses import dataclass, replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ecw_main_sim", os.path.join(HERE, "EnhancedCW 5 algo.py"))
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

# ------------------------------------------------------------------------------------------------
# constants shared with the threaded simulator (Appendix 1 of the manuscript)
# ------------------------------------------------------------------------------------------------
P_ACTIVE, P_IDLE, P_SLEEP, E_WAKE = sim.P_ACTIVE, sim.P_IDLE, sim.P_SLEEP, sim.E_WAKE
WAKE_THRESHOLD = sim.WAKE_THRESHOLD
UW_UNIT, UW_VERIFY = sim.USEFUL_WORK_COST, sim.USEFUL_WORK_VERIFY
SOL_B, ROUNDID_B, BLOCK_B = sim.CW_SOLUTION_MSG_BYTES, sim.CW_ROUNDID_MSG_BYTES, sim.CW_BLOCK_MSG_BYTES
COMMIT_B, REVEAL_B = sim.COMMIT_MSG_BYTES, sim.REVEAL_MSG_BYTES
VOTE_B = 100
UWVER_B = 32

# per-round success probabilities of the three targets used by the threaded simulator
P_HIT_ENH = 0x25 * 16 ** 60 / 2.0 ** 256      # Enhanced CW-PoW target: 1 771 expected hashes / round
P_HIT_BASIC = 0x9 * 16 ** 60 / 2.0 ** 256     # basic CW-PoW target:    7 282 expected hashes / round


@dataclass(frozen=True)
class Cfg:
    name: str = "Enhanced CW-PoW (all mechanisms, useful work)"
    n: int = 10                       # miners
    blocks: int = 30
    R0: int = sim.INITIAL_NBRR
    S0: int = sim.INITIAL_NBRS
    # --- mechanisms
    use_apt: bool = True
    use_cr: bool = True
    use_pipe: bool = True             # IDRound-independent part of the next header is precomputed
    pipe_overlap: str = "wait"        # where the precomputation runs: 'wait' = during the wait phase (first version),
                                      # 'hash' = on a separate unit while the previous round is being hashed
    wait_mode: str = "uw"             # 'uw' | 'sleep' | 'idle'
    use_committee: bool = True
    committee_size: int = sim.COMMITTEE_SIZE
    # --- work model
    p_hit: float = P_HIT_ENH          # per-attempt success probability of one round
    t_round: float = 0.020            # mean compute time of one round (s); sets tau = t_round * p_hit
    # --- costs (s), same names / values as the threaded ablation where they exist
    hdr_cost: float = 0.002           # full header assembly + mid-state per round (r > 1)
    hdr_fin_frac: float = 0.2         # phi: share of the header that depends on IDRound
    reveal_delay: float = 0.001       # d_rev: one reveal flight per round (commit-reveal)
    verify_cost: float = 0.001        # committee verification (active, committee members)
    vote_delay: float = 0.001         # vote flight
    uw_unit: float = UW_UNIT          # u: one useful-work unit
    uw_verify: float = UW_VERIFY      # verification of the useful-work proof per round
    t_wake: float = 0.001             # wake-up latency (duty cycling)
    net_delay: float = 0.0            # d: propagation delay of every broadcast
    # --- power model (units / s, units)
    p_active: float = P_ACTIVE
    p_idle: float = P_IDLE
    p_sleep: float = P_SLEEP
    e_wake: float = E_WAKE
    wake_threshold: float = WAKE_THRESHOLD
    # --- APT (rules of sim.adaptive_tuner; targets are parameters)
    apt_target_latency: float = sim.TARGET_LATENCY_PER_BLOCK
    apt_target_energy: float = sim.TARGET_ENERGY_PER_ROUND
    apt_target_msgs: int = sim.TARGET_MSGS_PER_BLOCK
    apt_bounds: tuple = (sim.APT_NBRR_MIN, sim.APT_NBRR_MAX, sim.APT_NBRS_MIN, sim.APT_NBRS_MAX)
    apt_rule: str = "priority"        # 'priority' (latency first, proposed) | 'original' (independent rules, first version)
    # --- disturbances
    delay_schedule: object = None     # block index -> extra propagation delay (s)
    spike_prob: float = 0.0           # probability that a broadcast is delayed ...
    spike_dur: float = 0.0            # ... by this many seconds
    n_withhold: int = 0               # miners that commit but never reveal
    reveal_timeout: float = 0.010     # tau_rev: reveal-or-forfeit timeout (s); None = wait forever
    final_quorum: bool = False        # block completes when S final candidates are visible (instead of all N)
    speeds: object = None             # optional per-miner hash-power multipliers (tuple of length n)
    sync_last: bool = False           # the last waiting round uses a full quorum (common start of the final round)
    final_window: object = None       # W (s): final candidates found within W of the first one compete; None = first finisher
    winner_rule: str = "min_hash"    # 'min_hash' (uniform among the last-round participants) | 'first' (first finisher; tested and rejected)
    selective: bool = True            # protocol rule: a round closes when S solutions exist; only the miners with a solution
                                      # start the next round (together); the others stand down until the block is final


def basic_cfg(**kw):
    """Basic CW-PoW structure: no mechanism, fixed (R, S), idle waiting, same per-round target."""
    base = dict(name="CW-PoW (basic structure)", use_apt=False, use_cr=False, use_pipe=False,
                wait_mode="idle", use_committee=False)
    base.update(kw)
    return Cfg(**base)


# ------------------------------------------------------------------------------------------------
# one block
# ------------------------------------------------------------------------------------------------
def quorum_times(s, a, S):
    """Time at which each miner sees S solutions: its own (visible at s_i) and the others (visible at a_j)."""
    n = len(s)
    order = np.argsort(a, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    srt = a[order]

    def kth_others(k):
        if k <= 0:
            return np.full(n, -np.inf)
        if k > n - 1:
            return np.full(n, np.inf)
        return np.where(rank < k, srt[k], srt[k - 1])

    return np.minimum(np.maximum(s, kth_others(S - 1)), kth_others(S))


def simulate_block(cfg, R, S, rng, extra_delay=0.0):
    if cfg.selective:
        return simulate_block_sel(cfg, R, S, rng, extra_delay)
    n = cfg.n
    d = cfg.net_delay + extra_delay
    attempt = cfg.t_round * cfg.p_hit
    mode = cfg.wait_mode
    z = np.zeros(n)
    hash_t, hdr_t, pre_t, uw_t, uwver_t, ver_t = z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    idle_t, sleep_t, wakelat_t, n_wake, uw_units = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    t = z.copy()
    msgs = 0
    nbytes = 0
    wait_sum = 0.0
    wait_cnt = 0
    long_waits = 0
    if cfg.use_cr:
        msgs += n
        nbytes += n * COMMIT_B
    rounds_wait = []
    final_vis = None
    f = None
    for r in range(1, R + 1):
        if r > 1 and cfg.hdr_cost > 0:
            fin = cfg.hdr_cost * (cfg.hdr_fin_frac if cfg.use_pipe else 1.0)
            t = t + fin
            hdr_t += fin
        X = attempt * rng.geometric(cfg.p_hit, size=n)
        if cfg.speeds is not None:
            X = X / np.asarray(cfg.speeds)
        t = t + X
        hash_t += X
        s = t.copy()
        msgs += n
        nbytes += n * SOL_B
        spike = (rng.random(n) < cfg.spike_prob) * cfg.spike_dur if cfg.spike_prob > 0 else 0.0
        if r == R:
            f = s
            final_vis = f + d + spike
            msgs += n
            nbytes += n * BLOCK_B
            break
        a = s + d + spike
        q = quorum_times(s, a, n if (cfg.sync_last and r == R - 1) else S)
        # precomputation of the IDRound-independent header part: in the wait phase ('wait', active time that
        # overlaps the wait) or hidden behind the hashing of the previous round ('hash', no extra time or power)
        pre = (cfg.hdr_cost * (1.0 - cfg.hdr_fin_frac)
               if (cfg.use_pipe and cfg.pipe_overlap == "wait" and cfg.hdr_cost > 0) else 0.0)
        t0 = s + pre
        pre_t += pre
        need = np.maximum(q - t0, 0.0)
        wait_sum += float(need.sum())
        wait_cnt += n
        slept = np.zeros(n, dtype=bool)
        if mode == "uw":
            k = np.ceil(need / cfg.uw_unit - 1e-12)
            dur = k * cfg.uw_unit
            uw_t += dur
            uw_units += k
        elif mode == "idle":
            dur = need
            idle_t += need
        else:  # sleep
            th = np.minimum(need, cfg.wake_threshold)
            idle_t += th
            sleep_t += need - th
            slept = need > cfg.wake_threshold
            n_wake += slept
            dur = need
        end = t0 + dur
        # commit-reveal flight (timeout if a reveal is withheld)
        if cfg.use_cr:
            if cfg.n_withhold > 0:
                rv = cfg.reveal_timeout if cfg.reveal_timeout is not None else math.inf
            else:
                rv = cfg.reveal_delay
            if mode == "uw":
                uw_t += rv
            elif mode == "sleep":
                sleep_t += np.where(slept, rv, 0.0)
                idle_t += np.where(slept, 0.0, rv)
            else:
                idle_t += rv
            end = end + rv
            msgs += n - cfg.n_withhold
            nbytes += (n - cfg.n_withhold) * REVEAL_B
        if mode == "uw":
            uwver_t += cfg.uw_verify
            end = end + cfg.uw_verify
            msgs += n
            nbytes += n * UWVER_B
        if mode == "sleep":
            wl = np.where(slept, cfg.t_wake, 0.0)
            wakelat_t += wl
            end = end + wl
        msgs += n
        nbytes += n * ROUNDID_B
        if cfg.sync_last and r == R - 1:       # common opening time of the final round (barrier)
            gap = float(end.max()) - end
            if mode == "uw":
                uw_t += gap
            else:
                idle_t += gap
            end = end + gap
        t = end
    if cfg.final_quorum:
        T_all = float(np.sort(final_vis)[min(S, n) - 1])
    else:
        T_all = float(final_vis.max())
    committee_time = (cfg.verify_cost + cfg.vote_delay) if cfg.use_committee else 0.0
    winner_w = None
    if cfg.final_window is not None:
        hend = float(f.min()) + cfg.final_window
        T_all = max(T_all, hend + d)
    L = T_all + committee_time
    # --- tail: from the last submission until the block is finalised
    if cfg.final_window is not None:
        sp_ = np.ones(n) if cfg.speeds is None else np.asarray(cfg.speeds)
        rate = sp_ / cfg.t_round
        elig = f <= hend
        cnt = np.where(elig, 1 + rng.poisson(rate * np.maximum(hend - f, 0.0)), 0)
        winner_w = int(rng.choice(n, p=cnt / cnt.sum()))
        ext = np.maximum(hend - f, 0.0)
        hash_t += ext
        f = np.maximum(f, hend)
    tail = L - f
    hash_t -= np.maximum(f - L, 0.0)      # miners still hashing at finalisation stop at L
    tail = np.maximum(tail, 0.0)
    members = rng.choice(n, size=min(cfg.committee_size, n), replace=False)   # drawn in every variant (common random numbers)
    if cfg.use_committee:
        ver_t[members] += cfg.verify_cost
        tail[members] -= cfg.verify_cost
        msgs += len(members)
        nbytes += len(members) * VOTE_B
    if mode == "uw":
        uw_t += tail
        uw_units += tail / cfg.uw_unit
    elif mode == "idle":
        idle_t += tail
    else:
        th = np.minimum(tail, cfg.wake_threshold)
        idle_t += th
        sleep_t += tail - th
        n_wake += tail > cfg.wake_threshold
    act = hash_t + hdr_t + pre_t + uw_t + uwver_t + ver_t
    E_hash = cfg.p_active * hash_t
    E_ovh = cfg.p_active * (hdr_t + pre_t + uwver_t + ver_t)
    E_uw = cfg.p_active * uw_t
    E_waste = cfg.p_idle * (idle_t + wakelat_t) + cfg.p_sleep * sleep_t + cfg.e_wake * n_wake
    E = E_hash + E_ovh + E_uw + E_waste
    total_t = act + idle_t + sleep_t + wakelat_t
    # time conservation (each miner is powered during the whole block interval)
    if not np.allclose(total_t, L, rtol=1e-9, atol=1e-9):
        raise AssertionError("time accounting error: %g vs %g" % (float(total_t.mean()), L))
    D = act / L
    return {
        "L": L, "E": float(E.mean()), "E_hash": float(E_hash.mean()), "E_ovh": float(E_ovh.mean()),
        "E_uw": float(E_uw.mean()), "E_waste": float(E_waste.mean()), "D": float(D.mean()),
        "MB": msgs, "BB": nbytes, "R": R, "S": S, "UWu": float(uw_units.mean()),
        "wakes": float(n_wake.mean()), "wait_mean": wait_sum / max(wait_cnt, 1),
        "hashes": float(hash_t.sum() / attempt), "winner_first": int(np.argmin(f)) if winner_w is None else winner_w,
    }


def simulate_block_sel(cfg, R, S, rng, extra_delay=0.0):
    """Protocol rule of the manuscript (Section 4): in round r the miners hash; the round closes at the instant
    at which S solutions are visible.  Only the miners that have a solution at that instant start round r+1, all
    at the same time; the others stand down until the block is finalised.  The winner is drawn among the
    miners of the last round (lowest final hash = uniform).  Every miner is powered during [0, L]."""
    n = cfg.n
    d = cfg.net_delay + extra_delay
    attempt = cfg.t_round * cfg.p_hit
    mode = cfg.wait_mode
    sp = np.ones(n) if cfg.speeds is None else np.asarray(cfg.speeds, dtype=float)
    z = np.zeros(n)
    hash_t, hdr_t, pre_t, uw_t, uwver_t, ver_t = z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    idle_t, sleep_t, wakelat_t, n_wake, uw_units = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    stop = z.copy()                       # instant at which a miner leaves the protocol (its tail begins)
    msgs, nbytes = 0, 0
    wait_sum, wait_cnt = 0.0, 0
    if cfg.use_cr:
        msgs += n
        nbytes += n * COMMIT_B
    alive = np.arange(n)
    T = 0.0
    part1 = None
    for r in range(1, R + 1):
        m = len(alive)
        fin = 0.0
        if r > 1 and cfg.hdr_cost > 0:
            fin = cfg.hdr_cost * (cfg.hdr_fin_frac if cfg.use_pipe else 1.0)
            hdr_t[alive] += fin
        t0h = T + fin
        X = attempt * rng.geometric(cfg.p_hit, size=m) / sp[alive]
        s = t0h + X
        spike = (rng.random(m) < cfg.spike_prob) * cfg.spike_dur if cfg.spike_prob > 0 else np.zeros(m)
        a = s + d + spike
        k = min(S, m)
        order = np.argsort(a, kind="stable")
        Q = float(a[order[k - 1]])                       # the round closes: S solutions are visible
        keep, drop = order[:k], order[k:]
        gone = alive[drop]
        hash_t[gone] += Q - t0h                          # they hash until they learn that the round is closed
        stop[gone] = Q
        who = alive[keep]
        hash_t[who] += X[keep]
        msgs += k
        nbytes += k * SOL_B
        if part1 is None:
            part1 = who.copy()
        if r == R:
            first = int(alive[order[0]])
            stop[who] = s[keep]
            msgs += k
            nbytes += k * BLOCK_B
            alive = who
            break
        sk = s[keep]
        pre = (cfg.hdr_cost * (1.0 - cfg.hdr_fin_frac)
               if (cfg.use_pipe and cfg.pipe_overlap == "wait" and cfg.hdr_cost > 0) else 0.0)
        t0 = sk + pre
        pre_t[who] += pre
        need = np.maximum(Q - t0, 0.0)
        wait_sum += float(need.sum())
        wait_cnt += k
        slept = np.zeros(k, dtype=bool)
        if mode == "uw":
            ku = np.ceil(need / cfg.uw_unit - 1e-12)
            dur = ku * cfg.uw_unit
            uw_t[who] += dur
            uw_units[who] += ku
        elif mode == "idle":
            dur = need
            idle_t[who] += need
        else:
            th = np.minimum(need, cfg.wake_threshold)
            idle_t[who] += th
            sleep_t[who] += need - th
            slept = need > cfg.wake_threshold
            n_wake[who] += slept
            dur = need
        end = t0 + dur
        if cfg.use_cr:
            if cfg.n_withhold > 0:
                rv = cfg.reveal_timeout if cfg.reveal_timeout is not None else math.inf
            else:
                rv = cfg.reveal_delay
            if mode == "uw":
                uw_t[who] += rv
            elif mode == "sleep":
                sleep_t[who] += np.where(slept, rv, 0.0)
                idle_t[who] += np.where(slept, 0.0, rv)
            else:
                idle_t[who] += rv
            end = end + rv
            msgs += k - cfg.n_withhold
            nbytes += (k - cfg.n_withhold) * REVEAL_B
        if mode == "uw":
            uwver_t[who] += cfg.uw_verify
            end = end + cfg.uw_verify
            msgs += k
            nbytes += k * UWVER_B
        if mode == "sleep":
            wl = np.where(slept, cfg.t_wake, 0.0)
            wakelat_t[who] += wl
            end = end + wl
        msgs += k
        nbytes += k * ROUNDID_B
        # common start of the next round: the miners that reached it earlier wait for the others
        Tn = float(end.max())
        gap = Tn - end
        if mode == "uw":
            uw_t[who] += gap
        else:
            idle_t[who] += gap
        T = Tn
        alive = who
    committee_time = (cfg.verify_cost + cfg.vote_delay) if cfg.use_committee else 0.0
    L = Q + committee_time
    winner = first if cfg.winner_rule == "first" else int(rng.choice(alive))
    tail = L - stop
    members = rng.choice(n, size=min(cfg.committee_size, n), replace=False)
    if cfg.use_committee:
        ver_t[members] += cfg.verify_cost
        tail[members] -= cfg.verify_cost
        msgs += len(members)
        nbytes += len(members) * VOTE_B
    if mode == "uw":
        uw_t += tail
        uw_units += tail / cfg.uw_unit
    elif mode == "idle":
        idle_t += tail
    else:
        th = np.minimum(tail, cfg.wake_threshold)
        idle_t += th
        sleep_t += tail - th
        n_wake += tail > cfg.wake_threshold
    act = hash_t + hdr_t + pre_t + uw_t + uwver_t + ver_t
    E_hash = cfg.p_active * hash_t
    E_ovh = cfg.p_active * (hdr_t + pre_t + uwver_t + ver_t)
    E_uw = cfg.p_active * uw_t
    E_waste = cfg.p_idle * (idle_t + wakelat_t) + cfg.p_sleep * sleep_t + cfg.e_wake * n_wake
    E = E_hash + E_ovh + E_uw + E_waste
    total_t = act + idle_t + sleep_t + wakelat_t
    if not np.allclose(total_t, L, rtol=1e-9, atol=1e-9):
        raise AssertionError("time accounting error: %g vs %g" % (float(total_t.mean()), L))
    D = act / L
    return {
        "L": L, "E": float(E.mean()), "E_hash": float(E_hash.mean()), "E_ovh": float(E_ovh.mean()),
        "E_uw": float(E_uw.mean()), "E_waste": float(E_waste.mean()), "D": float(D.mean()),
        "MB": msgs, "BB": nbytes, "R": R, "S": S, "UWu": float(uw_units.mean()),
        "wakes": float(n_wake.mean()), "wait_mean": wait_sum / max(wait_cnt, 1),
        "hashes": float(hash_t.sum() / attempt), "winner_first": winner, "part1": part1,
    }


def apt_update(cfg, R, S, obs):
    r_min, r_max, s_min, s_max = cfg.apt_bounds
    nR, nS = R, S
    if cfg.apt_rule == "priority":
        # latency first: the energy rule is only consulted inside the latency band
        if obs["lat"] > cfg.apt_target_latency:
            nR = max(r_min, nR - 1)
        elif obs["lat"] < 0.5 * cfg.apt_target_latency:
            nR = min(r_max, nR + 1)
        elif obs["e_round"] > cfg.apt_target_energy:
            nR = min(r_max, nR + 1)
    else:
        # rules of the threaded simulator (sim.adaptive_tuner): independent, the energy rule always fires
        if obs["lat"] > cfg.apt_target_latency:
            nR = max(r_min, nR - 1)
        elif obs["lat"] < 0.5 * cfg.apt_target_latency:
            nR = min(r_max, nR + 1)
        if obs["e_round"] > cfg.apt_target_energy:
            nR = min(r_max, nR + 1)
    if obs["msgs"] > cfg.apt_target_msgs:
        nS = min(s_max, nS + 1)
    elif obs["msgs"] < 0.5 * cfg.apt_target_msgs:
        nS = max(s_min, nS - 1)
    nR = max(r_min, min(r_max, int(nR)))
    nS = max(s_min, min(s_max, int(nS)))
    if abs(nR - R) > 1:
        nR = R + (1 if nR > R else -1)
    if abs(nS - S) > 1:
        nS = S + (1 if nS > S else -1)
    return nR, nS


def run(cfg, seed):
    """Simulate cfg.blocks consecutive blocks; return the list of per-block records."""
    rng = np.random.default_rng(seed)
    R, S = cfg.R0, cfg.S0
    out = []
    for b in range(cfg.blocks):
        extra = cfg.delay_schedule(b) if cfg.delay_schedule else 0.0
        rec = simulate_block(cfg, R, S, rng, extra)
        out.append(rec)
        if cfg.use_apt:
            R, S = apt_update(cfg, R, S, {"lat": rec["L"], "e_round": rec["E"] / max(R, 1), "msgs": rec["MB"]})
    return out


def pow_block(n, work, t_attempt, d_net, rng, p_active=P_ACTIVE):
    """Classical PoW race with the same hardware model: `work` = expected hashes of the whole network
    per block (every miner hashes at 1/t_attempt until the block of the winner reaches it)."""
    p = 1.0 / work
    T = t_attempt * rng.geometric(p, size=n)
    Tw = float(T.min())
    L = Tw + d_net
    return {"L": L, "E": p_active * L, "E_hash": p_active * L, "D": 1.0, "MB": n * 0 + 1, "BB": 1_000_000, "hashes": n * L / t_attempt}


# ------------------------------------------------------------------------------------------------
# reference protocols of Table 1 (same hardware and power model; per-block records)
# ------------------------------------------------------------------------------------------------
TAU = 0.020 * P_HIT_ENH                  # seconds per hash attempt of one miner (calibrated: 20 ms / 1 771 hashes)
P_POW = sim.POW_TARGET / 2.0 ** 256      # per-attempt success probability of the PoW / Green-PoW / PoE target
POE_RELAX = 2e-6                         # eased-target credit per hash (sim.POE_RELAX_PER_HASH / POW_TARGET)


def pow_blocks(n, blocks, rng, p=P_POW, tau=TAU, d=0.0):
    out = []
    for _ in range(blocks):
        Tw = float((tau * rng.geometric(p, size=n)).min())
        L = Tw + d
        out.append({"L": L, "E": P_ACTIVE * L, "D": 1.0, "MB": 1, "BB": 1_000_000, "W": n * L / tau})
    return out


def green_blocks(n, blocks, rng, p=P_POW, tau=TAU, d=0.0):
    """Open round (all race) followed by an exclusive round of the runner-up (others idle), alternating
    (rule of sim.run_green_pow: a solo exclusive round returns to open competition)."""
    out = []
    exclusive = False
    for _ in range(blocks):
        if not exclusive:
            Tw = float((tau * rng.geometric(p, size=n)).min())
            L = Tw + d
            out.append({"L": L, "E": P_ACTIVE * L, "D": 1.0, "MB": 1, "BB": 1_000_000, "W": n * L / tau})
        else:
            Tx = float(tau * rng.geometric(p))
            L = Tx + d
            act = Tx
            E = (P_ACTIVE * act + P_IDLE * (n * L - act)) / n
            out.append({"L": L, "E": E, "D": act / (n * L), "MB": 1, "BB": 1_000_000, "W": Tx / tau})
        exclusive = not exclusive
    return out


def poe_blocks(n, blocks, rng, p=P_POW, tau=TAU, d=0.0, relax=POE_RELAX):
    """Every round is open; the target of a miner is eased in proportion to its unsuccessful hashes
    since its last win (rule of sim.run_proof_of_experience)."""
    out = []
    credit = np.zeros(n)
    for _ in range(blocks):
        pi = np.minimum(p * (1.0 + credit), 1.0)
        T = tau * rng.geometric(pi)
        i = int(T.argmin())
        L = float(T[i]) + d
        h = L / tau
        out.append({"L": L, "E": P_ACTIVE * L, "D": 1.0, "MB": 1, "BB": 1_000_000, "W": n * h})
        credit += relax * h
        credit[i] = 0.0
    return out


def pos_blocks(blocks, validators=50, committee=11, t_prop=0.001, t_vote=0.002):
    """Prototype rule: one proposal, then the committee votes are processed one after the other."""
    L = t_prop + committee * t_vote
    act = t_prop + committee * t_vote
    E = (P_ACTIVE * act + P_IDLE * (validators * L - act)) / validators
    return [{"L": L, "E": E, "D": act / (validators * L), "MB": 1 + committee, "BB": 2000 + committee * 200, "W": 0.0}
            for _ in range(blocks)]


def pbft_blocks(blocks, members=13, t_msg=0.001):
    """Prototype rule: pre-prepare (1 message), prepare and commit of every member, processed in sequence."""
    L = t_msg * (1 + 2 * members)
    act = L
    E = (P_ACTIVE * act + P_IDLE * (members * L - act)) / members
    return [{"L": L, "E": E, "D": act / (members * L), "MB": 1 + 2 * members,
             "BB": 1000 + members * 1000 + members * 500, "W": 0.0} for _ in range(blocks)]


# ------------------------------------------------------------------------------------------------
# statistics helpers
# ------------------------------------------------------------------------------------------------
KEYS = ["L", "E", "E_hash", "E_ovh", "E_uw", "E_waste", "D", "MB", "BB", "R", "S", "UWu", "wakes", "wait_mean"]


def steady(rows, skip):
    """Mean of each key over the blocks after the first `skip` blocks of one repetition."""
    sel = rows[skip:] if len(rows) > skip else rows
    return {k: float(np.mean([r[k] for r in sel])) for k in KEYS}


def replicate(cfg, reps, skip=0, seed0=1000):
    return [steady(run(cfg, seed0 + k), skip) for k in range(reps)]


def summarize(reps_rows):
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in reps_rows])
        m = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[k] = (m, sd, 1.96 * sd / math.sqrt(len(v)) if len(v) > 1 else 0.0)
    return out


def write_csv(name, header, rows):
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    print("ecw_des.py is the model; the experiments of the manuscript are in ecw_des_experiments.py")
