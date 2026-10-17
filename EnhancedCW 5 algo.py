"""
consensus_simulators.py

Provides three simple simulators (Enhanced CW-PoW, Basic CW-PoW, PoW, PoS, PBFT) for comparative experiments.
Outputs CSV summary files and prints throughput / energy / fairness metrics.

NOTE: This is a lightweight *experimental* simulator (not a full network simulator).
It's designed for quick, repeatable experiments to compare high-level metrics.

Unified simulator with:
 - Enhanced CW-PoW
 - Basic CW-PoW 
 - Standard PoW
 - Simple PoS
 - Simple PBFT

"""

import threading
import random
import hashlib
import time
import csv
from math import ceil

# -----------------------
# GLOBAL CONSTANTS
# -----------------------
POWER_ACTIVE = 100.0     # energy units/sec (active hashing / computation)
POWER_WAIT = 5.0         # energy units/sec (idle/wait)
SLEEP_POLL = 0.001

# Message size assumptions (bytes)
CW_SOLUTION_MSG_BYTES = 1000        # per-round solution broadcast (1 KB)
CW_ROUNDID_MSG_BYTES = 32           # round identifier broadcast
CW_BLOCK_MSG_BYTES = 2000           # final block broadcast (2 KB)

POW_BLOCK_MSG_BYTES = 1000000       # PoW block broadcast (1 MB)

POS_PROPOSE_BYTES = 2000
POS_VOTE_BYTES = 200

PBFT_PREPARE_BYTES = 1000
PBFT_COMMIT_BYTES = 500

# -----------------------
# UTILITIES
# -----------------------
def now():
    return time.perf_counter()

def write_csv(filename, header, rows):
    with open(filename, "w", newline="") as f:
        import csv as _csv
        writer = _csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

# -------------------------
# 1) enhanced CW
# -------------------------
# -------------------------
# Configuration (tuneable)
# -------------------------
NUM_MINERS = 10
INITIAL_NBRR = 6        # initial number of rounds per block
INITIAL_NBRS = 3        # initial number of solutions required per round
RUNS = 5               # number of blocks to simulate (for fairness/stats)
TARGET_BITS = 22        # difficulty parameter (coarse)
COMMIT_VDF_DELAY = 0.0  # simulate VDF delay (seconds) if desired
COMMIT_REVEAL = True

# Useful work substitution
USE_USEFUL_WORK = True
USEFUL_WORK_COST = 0.005   # seconds of active compute simulated per useful-work task unit
USEFUL_WORK_VERIFY = 0.001 # verification cost per useful-work (seconds)

# Duty cycling / energy
P_ACTIVE = 100.0      # energy units/sec during active hashing/useful work
P_IDLE = 5.0          # energy units/sec while idle-but-awake
P_SLEEP = 0.5         # energy units/sec while sleeping (duty-cycling)
E_WAKE = 1.0          # energy units cost to wake device (units)
WAKE_THRESHOLD = 0.001# minimal wait period to actually sleep

# Message sizes (bytes)
CW_SOLUTION_MSG_BYTES = 1000
CW_ROUNDID_MSG_BYTES = 32
CW_BLOCK_MSG_BYTES = 2000
COMMIT_MSG_BYTES = 64
REVEAL_MSG_BYTES = 64

# APT (Adaptive Parameter Tuning) targets & gains (simple controller)
TARGET_LATENCY_PER_BLOCK = 1.0     # seconds desirable per block (tunable)
TARGET_ENERGY_PER_ROUND = 2.0      # units desirable per miner per round
TARGET_MSGS_PER_BLOCK = 200        # desired bound on messages per block
APT_NBRR_MIN = 2
APT_NBRR_MAX = 12
APT_NBRS_MIN = 1
APT_NBRS_MAX = 6

# Hybrid committee
COMMITTEE_SIZE = 7

# Other
SLEEP_POLL = 0.001
SEED = 2025

# -------------------------
# APT: simple heuristic controller
# -------------------------
def adaptive_tuner(curr_NbrR, curr_NbrS, observed):
    """
    observed: dict with keys:
      - avg_latency_per_block (s)
      - avg_energy_per_round (units)
      - msgs_per_block (count)
    Heuristic:
      - If latency too high, decrease NbrR (down to min)
      - If energy per round too high, increase NbrR (up to max)
      - If msgs per block too high, increase NbrS (reduces round churn)
    Returns updated (NbrR, NbrS)
    """
    NbrR = curr_NbrR
    NbrS = curr_NbrS

    # Latency control
    if observed["avg_latency_per_block"] > TARGET_LATENCY_PER_BLOCK:
        NbrR = max(APT_NBRR_MIN, NbrR - 1)
    else:
        # if latency is comfortably below target, we can try increase rounds (save energy)
        if observed["avg_latency_per_block"] < 0.5 * TARGET_LATENCY_PER_BLOCK:
            NbrR = min(APT_NBRR_MAX, NbrR + 1)

    # Energy control
    if observed["avg_energy_per_round"] > TARGET_ENERGY_PER_ROUND:
        NbrR = min(APT_NBRR_MAX, NbrR + 1)

    # Communication control
    if observed["msgs_per_block"] > TARGET_MSGS_PER_BLOCK:
        NbrS = min(APT_NBRS_MAX, NbrS + 1)
    elif observed["msgs_per_block"] < TARGET_MSGS_PER_BLOCK * 0.5:
        NbrS = max(APT_NBRS_MIN, NbrS - 1)

    # Ensure bounds
    NbrR = max(APT_NBRR_MIN, min(APT_NBRR_MAX, int(NbrR)))
    NbrS = max(APT_NBRS_MIN, min(APT_NBRS_MAX, int(NbrS)))
    return NbrR, NbrS

# -------------------------
# Enhanced CW-PoW simulator
# -------------------------
def run_enhanced_cw_pow(num_miners=NUM_MINERS,
                        nbrR=INITIAL_NBRR,
                        nbrS=INITIAL_NBRS,
                        runs=RUNS,
                        target_bits=TARGET_BITS,
                        committee_size=COMMITTEE_SIZE,
                        seed=SEED,
                        byzantine_withhold_frac=0.0,
                        reveal_timeout_polls=None,
                        network_delay_spike_prob=0.0,
                        network_delay_spike_sec=0.0):
    """
    byzantine_withhold_frac: fraction of miners (in [0,1)) that compute a valid
        solution but withhold their final broadcast/reveal every round, modeling
        Byzantine solution-withholding behavior.
    reveal_timeout_polls: if set, bounds the number of quorum-polling iterations
        a miner performs before proceeding with whichever reveals already
        arrived (reveal-or-forfeit liveness fallback). None reproduces the
        original (unbounded) behavior used to obtain the Table 1 results.
    network_delay_spike_prob / network_delay_spike_sec: with the given
        probability, a broadcast experiences an extra delay spike of the given
        duration, modeling bursty network conditions.
    """
    random.seed(seed)
    #TARGET = (1 << (256 - target_bits)) - 1
    TARGET = int("25000000000000000000000000000000000000000000000000000000000000", 16)

    # accumulators across runs
    total_compute = [0.0] * num_miners
    total_wait = [0.0] * num_miners
    total_hashes = [0] * num_miners
    total_msgs = [0] * num_miners
    total_bytes = [0] * num_miners
    wins = [0] * num_miners

    # metrics for APT
    observed_history = []

    lock = threading.Lock()

    # initial parameters (will be adapted)
    cur_NbrR = nbrR
    cur_NbrS = nbrS

    for run_id in range(runs):
        # per-run structures
        queues = [ [] for _ in range(cur_NbrR) ]   # list of (miner_id, nonce)
        commits = [None] * num_miners
        reveals = [None] * num_miners
        final_nonce = [None] * num_miners
        final_hash = [None] * num_miners

        # per-run per-miner counters
        run_compute = [0.0] * num_miners
        run_wait = [0.0] * num_miners
        run_hashes = [0] * num_miners
        run_msgs = [0] * num_miners
        run_bytes = [0] * num_miners

        # committee (random subset)
        committee = random.sample(range(num_miners), min(committee_size, num_miners))

        # Byzantine withholders: miners that commit but never reveal (liveness/robustness test)
        n_withholders = int(byzantine_withhold_frac * num_miners)
        withholders = set(random.sample(range(num_miners), n_withholders)) if n_withholders > 0 else set()

        # Commit phase: miners broadcast commitments (if using C-R)
        if COMMIT_REVEAL:
            for i in range(num_miners):
                # miner chooses random r_i and commits (we only simulate sizes + msgs)
                r_i = random.getrandbits(128)
                commits[i] = r_i  # save locally
                run_msgs[i] += 1
                run_bytes[i] += COMMIT_MSG_BYTES
            # (optionally simulate VDF on aggregate, but small delay)
            if COMMIT_VDF_DELAY > 0:
                time.sleep(COMMIT_VDF_DELAY)
            # withholders commit but never reveal: their commitment is dropped from
            # the aggregate seed, modeling the "commits but does not reveal" case
            for i in withholders:
                commits[i] = None

        # per-miner worker
        def miner_proc(i):
            rng = random.Random(seed + run_id*1000 + i*17)
            x = rng.randint(10_000, 10**18)
            nonce = 1
            round_id = 0

            # reveal occurs at end of previous block's commit stage simulated above
            # here we will reveal current randomness at the moment needed
            for r in range(cur_NbrR):
                # COMPUTE PHASE (hashing)
                t0 = now()
                h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                run_hashes[i] += 1
                while h > TARGET:
                    nonce += 1
                    h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                    run_hashes[i] += 1
                t1 = now()
                run_compute[i] += (t1 - t0)

                # Broadcast solution message
                with lock:
                    queues[r].append((i, nonce))
                run_msgs[i] += 1
                run_bytes[i] += CW_SOLUTION_MSG_BYTES

                # Bursty network condition: occasional propagation delay spike
                if network_delay_spike_prob > 0 and rng.random() < network_delay_spike_prob:
                    spike0 = now()
                    time.sleep(network_delay_spike_sec)
                    run_wait[i] += (now() - spike0)

                # If last round -> final candidate broadcast (to network and committee)
                if r == cur_NbrR - 1:
                    final_nonce[i] = nonce
                    final_hash[i] = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                    run_msgs[i] += 1
                    run_bytes[i] += CW_BLOCK_MSG_BYTES
                    break

                # WAIT PHASE: either sleep (duty-cycle) or do useful work or idle
                t_wait_start = now()

                # Optionally go to sleep if wait expected > threshold
                # here we don't know exact wait length; we simulate simple decision: if useful work enabled -> do useful work
                if USE_USEFUL_WORK:
                    # simulate doing a bounded amount of useful work while waiting
                    # the miner performs useful units until quorum reached, or until
                    # reveal_timeout_polls is exhausted (reveal-or-forfeit liveness fallback)
                    # we simulate it as small loop doing useful-work calls and periodic checks
                    polls = 0
                    while True:
                        # check quorum
                        with lock:
                            quorum_reached = (len(queues[r]) >= cur_NbrS)
                        if quorum_reached:
                            break
                        polls += 1
                        if reveal_timeout_polls is not None and polls >= reveal_timeout_polls:
                            break  # timed out: proceed with whichever reveals have arrived
                        # perform one useful-work unit (active)
                        uw0 = now()
                        # simulate CPU cost
                        time.sleep(USEFUL_WORK_COST)  # active CPU doing useful work
                        run_compute[i] += (now() - uw0)
                        # generate a small proof and broadcast verification-size message (cheap)
                        run_msgs[i] += 0  # we assume useful-work results are not broadcast every unit
                        # optional quick verify overhead when finishing
                    # after loop, do cheap verification emission
                    run_msgs[i] += 1
                    run_bytes[i] += USEFUL_WORK_VERIFY * 1000  # approximate
                else:
                    # Duty-cycling: sleep until quorum reached, or until timeout (liveness fallback)
                    slept = False
                    polls = 0
                    while True:
                        with lock:
                            quorum_reached = (len(queues[r]) >= cur_NbrS)
                        if quorum_reached:
                            break
                        polls += 1
                        if reveal_timeout_polls is not None and polls >= reveal_timeout_polls:
                            break
                        # Sleep for a short time to emulate low-power mode
                        time.sleep(SLEEP_POLL)
                        slept = True
                    if slept:
                        # pay wake-up energy cost accounted later per-run
                        pass

                t_wait_end = now()
                run_wait[i] += (t_wait_end - t_wait_start)

                # Compute roundID from aggregate commit-reveal + first cur_NbrS solutions
                # Reveal stage: miners broadcast their reveal once quorum reached (simulate).
                # Withholding miners (i in withholders) do not send a reveal message, matching
                # the "commits but does not reveal" case and the reveal-or-forfeit rule.
                if COMMIT_REVEAL and i not in withholders:
                    # reveal r_i (we don't use all reveals explicitly, just simulate msgs/bytes)
                    run_msgs[i] += 1
                    run_bytes[i] += REVEAL_MSG_BYTES

                with lock:
                    # sum first cur_NbrS nonces deterministically (if not enough entries, sum what's there)
                    available = len(queues[r])
                    take = min(cur_NbrS, available)
                    s = sum([queues[r][k][1] for k in range(take)]) if take > 0 else 0
                # incorporate commit randomness into seed if present
                if COMMIT_REVEAL:
                    # take aggregate of reveal values (simple XOR sum) to simulate seed
                    agg = 0
                    for j in range(num_miners):
                        if commits[j] is not None:
                            agg ^= commits[j]  # combine commits (simple model)
                    seed_val = (agg + s) & ((1 << 256) - 1)
                else:
                    seed_val = s

                round_id = seed_val  # set next round identifier
                # broadcast roundID
                run_msgs[i] += 1
                run_bytes[i] += CW_ROUNDID_MSG_BYTES

                # precompute middlestate (simulated): small time saving on next round
                # We'll simulate precompute cost and assume it reduces next compute time by a factor
                # but for simplicity we just note it (no change to hashing loop here)
                nonce = 1

        # start miner threads for this run
        threads = []
        for i in range(num_miners):
            t = threading.Thread(target=miner_proc, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # choose winner among final_hash (lowest)
        winner = None
        min_val = None
        for i in range(num_miners):
            if final_hash[i] is None:
                continue
            if min_val is None or final_hash[i] < min_val:
                min_val = final_hash[i]
                winner = i
        if winner is not None:
            wins[winner] += 1

        # Hybrid committee finalization: committee verifies the winner
        # simulate committee verification + votes
        # each committee member verifies (small compute) and casts vote; need 2/3 votes
        votes = 0
        for m in committee:
            # simulate verify cost
            time.sleep(0.001)
            run_compute[m] += 0.001
            run_msgs[m] += 1
            run_bytes[m] += 100  # vote message small
            votes += 1
        if votes >= ceil(2*len(committee)/3):
            # finalized (for simulation, always succeed because no adversary modeled)
            pass

        # accumulate run stats into totals
        for i in range(num_miners):
            total_compute[i] += run_compute[i]
            total_wait[i] += run_wait[i]
            total_hashes[i] += run_hashes[i]
            total_msgs[i] += run_msgs[i]
            total_bytes[i] += run_bytes[i]

        # After each run (block) we collect observed metrics and let APT update parameters
        # compute observed values for this run
        # avg latency per block approximated by average (compute+wait) across miners
        total_times_run = [run_compute[i] + run_wait[i] for i in range(num_miners)]
        avg_latency_run = sum(total_times_run) / len(total_times_run)
        # avg energy per round (approx): (compute_time * P_ACTIVE + wait_time * P_IDLE) / nbrR
        energy_run_total = sum([run_compute[i]*P_ACTIVE + run_wait[i]*P_IDLE for i in range(num_miners)])
        avg_energy_per_round = (energy_run_total / num_miners) / max(1, cur_NbrR)
        msgs_this_run = sum(run_msgs)
        observed = {
            "avg_latency_per_block": avg_latency_run,
            "avg_energy_per_round": avg_energy_per_round,
            "msgs_per_block": msgs_this_run
        }
        observed_history.append(observed)

        # Adapt parameters for next block
        new_NbrR, new_NbrS = adaptive_tuner(cur_NbrR, cur_NbrS, observed)
        # For stability, only change by at most 1
        if abs(new_NbrR - cur_NbrR) > 1:
            new_NbrR = cur_NbrR + (1 if new_NbrR > cur_NbrR else -1)
        if abs(new_NbrS - cur_NbrS) > 1:
            new_NbrS = cur_NbrS + (1 if new_NbrS > cur_NbrS else -1)
        cur_NbrR, cur_NbrS = new_NbrR, new_NbrS

    # After all runs: compute fairness, energy, throughput, etc.
    total_hash_sum = sum(total_hashes) or 1
    rows = []
    fairness_list = []
    energies = []
    efficiencies = []
    total_times = []
    for i in range(num_miners):
        p_win = wins[i] / runs
        resource_share = total_hashes[i] / total_hash_sum
        F_i = (p_win / resource_share) if resource_share > 0 else 0.0
        fairness_list.append(F_i)
        energy_i = total_compute[i]*P_ACTIVE + total_wait[i]*P_IDLE + (runs * E_WAKE) # wake cost approx
        energies.append(energy_i)
        total_time_i = total_compute[i] + total_wait[i]
        total_times.append(total_time_i)
        efficiencies.append(1 - (total_wait[i] / total_time_i))

        rows.append([
            i+1,
            total_compute[i],
            total_wait[i],
            total_hashes[i],
            wins[i],
            total_msgs[i],
            total_bytes[i],
            p_win,
            resource_share,
            F_i,
            energy_i,
            total_time_i
        ])

    avg_fairness = sum(fairness_list) / len(fairness_list)
    # normalized avg energy per miner per round across runs
    avg_energy = sum(energies) / (num_miners * cur_NbrR * runs)
    avg_total_time = sum(total_times) / len(total_times)
    msgs_per_block = sum(total_msgs) / runs
    bytes_per_block = sum(total_bytes) / runs
    # throughput: runs blocks produced over approx avg_total_time (per miner). Estimate system throughput:
    T_experiment = avg_total_time if avg_total_time > 0 else 1.0
    throughput = runs / T_experiment
    avg_efficiency = sum(efficiencies) / len(efficiencies)
      

    print("=== Enhanced CW-PoW Summary ===")
    print(f"Initial params (NbrR, NbrS): ({nbrR},{nbrS})")
    print(f"Final params after adaptation (NbrR, NbrS): ({cur_NbrR},{cur_NbrS})")
    #print(f"Runs (blocks): {runs}")
    print(f"Avg total time per miner (s): {avg_total_time:.4f}")
    print(f"Throughput (blocks/s, est): {throughput:.4f}")
    print(f"Avg fairness (mean F_i): {avg_fairness:.4f}")
    print(f"Messages per block (total net): {msgs_per_block:.2f}")
    print(f"Bytes per block (approx): {bytes_per_block:.1f}")
    print(f"Avg energy per miner per round: {avg_energy:.4f} units")
    print(f"Active-Time Fraction (E): {avg_efficiency:.3f}")

    header = [
        "Miner","Compute_total","Wait_total","Hashes_total","Wins",
        "Msgs_total","Bytes_total","P_win","ResShare","F_i","Energy","Total_time"
    ]
    write_csv("enhanced_cw_pow_results.csv", header, rows)

    return {
        "avg_fairness": avg_fairness,
        "avg_total_time": avg_total_time,
        "throughput": throughput,
        "msgs_per_block": msgs_per_block,
        "bytes_per_block": bytes_per_block,
        "avg_energy_per_round": avg_energy,
        "final_params": (cur_NbrR, cur_NbrS),
        "observed_history": observed_history
    }

# -----------------------
# 2) CW-PoW (multi-run fairness + comm counting)
# -----------------------
def run_cw_pow(num_miners=NUM_MINERS, last_round=5, sol_nb=3, runs=RUNS, target_bits=20, seed=1234):
    """
    CW-PoW simulation:
    - For each 'run' we simulate one block's sequence of rounds (NbrR = last_round).
    - Each miner performs rounds, broadcasts per-round solutions and roundIDs.
    - At the final round, miners broadcast final block candidate; we pick the 'winner'
      as the miner with the minimal final hash (tie-break deterministic).
    - We repeat 'runs' times to estimate P_win for each miner and compute fairness.
    - Communication metrics: counts of messages and bytes (solution, roundID, final block).
    """
    random.seed(seed)
    TARGET = int("9000000000000000000000000000000000000000000000000000000000000", 16)
#(1 << (256 - target_bits)) - 1

    # global accumulators across runs
    total_compute = [0.0] * num_miners
    total_wait = [0.0] * num_miners
    total_hashes = [0] * num_miners
    total_msgs = [0] * num_miners
    total_bytes = [0] * num_miners
    wins = [0] * num_miners

    lock = threading.Lock()
    
    
    for run_id in range(runs):
        # per-run structures
        queues = [ [] for _ in range(last_round) ]   # each element: list of (miner_id, nonce)
        final_nonce = [None] * num_miners
        final_hashval = [None] * num_miners

        # per-run per-miner accumulators
        run_compute = [0.0] * num_miners
        run_wait = [0.0] * num_miners
        run_hashes = [0] * num_miners
        run_msgs = [0] * num_miners
        run_bytes = [0] * num_miners

        def miner_proc(i):
            rng = random.Random(seed + run_id*1000 + i*37)
            nonce = 1
            round_id = 0
            x = rng.randint(10_000, 10**18)

            for r in range(last_round):
                # COMPUTE: keep hashing until <= TARGET
                t0 = now()
                h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                run_hashes[i] += 1
                while h > TARGET:
                    nonce += 1
                    h = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                    run_hashes[i] += 1
                t1 = now()
                run_compute[i] += (t1 - t0)

                # Broadcast solution (one message per miner per round)
                with lock:
                    queues[r].append((i, nonce))
                run_msgs[i] += 1
                run_bytes[i] += CW_SOLUTION_MSG_BYTES

                # If final round, broadcast final block and exit
                if r == last_round - 1:
                    # final candidate broadcast
                    # compute final hashval for tie-breaking
                    final_nonce[i] = nonce
                    final_hashval[i] = int(hashlib.sha256(f"{nonce}{round_id}{x}".encode()).hexdigest(), 16)
                    run_msgs[i] += 1    # final block broadcast
                    run_bytes[i] += CW_BLOCK_MSG_BYTES
                    break

                # WAIT: wait until quorum of solutions are present
                t_wait_start = now()
                while True:
                    with lock:
                        if len(queues[r]) >= sol_nb:
                            break
                    time.sleep(SLEEP_POLL)
                t_wait_end = now()
                run_wait[i] += (t_wait_end - t_wait_start)

                # Compute roundID from first sol_nb solutions (sum nonces) and broadcast it
                with lock:
                    # sum nonces of the first sol_nb entries
                    s = sum([queues[r][k][1] for k in range(sol_nb)])
                round_id = s
                # broadcast IDRound (we count each miner broadcasting an IDRound once per round)
                run_msgs[i] += 1
                run_bytes[i] += CW_ROUNDID_MSG_BYTES

                nonce = 1  # reset nonce for next round

        # start miner threads for this run
        threads = []
        for i in range(num_miners):
            t = threading.Thread(target=miner_proc, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # choose winner: minimal final_hashval (deterministic tie-break)
        min_val = None
        winner = None
        for i in range(num_miners):
            if final_hashval[i] is None:
                continue
            if min_val is None or final_hashval[i] < min_val:
                min_val = final_hashval[i]
                winner = i
        if winner is not None:
            wins[winner] += 1

        # accumulate run stats into totals
        for i in range(num_miners):
            total_compute[i] += run_compute[i]
            total_wait[i] += run_wait[i]
            total_hashes[i] += run_hashes[i]
            total_msgs[i] += run_msgs[i]
            total_bytes[i] += run_bytes[i]

    
    
    
    # After all runs: compute fairness + timing + throughput
    total_hash_sum = sum(total_hashes) or 1
    rows = []
    fairness_list = []
    energies = []
    efficiencies = []
    total_times = []   # compute + wait
    
    for i in range(num_miners):
        p_win = wins[i] / runs
        resource_share = total_hashes[i] / total_hash_sum
        F_i = (p_win / resource_share) if resource_share > 0 else 0.0
        fairness_list.append(F_i)
    
        # energy and total time
        energy = total_compute[i] * POWER_ACTIVE + total_wait[i] * POWER_WAIT
        total_time_i = total_compute[i] + total_wait[i]
        total_times.append(total_time_i)
        energies.append(energy)
        efficiencies.append(1 - (total_wait[i] / total_time_i))

        rows.append([
            i + 1,
            total_compute[i],
            total_wait[i],
            total_hashes[i],
            wins[i],
            total_msgs[i],
            total_bytes[i],
            p_win,
            resource_share,
            F_i,
            energy,
            total_time_i
        ])
    
    # --- Averages ---
    avg_fairness = sum(fairness_list) / len(fairness_list)
    avg_energy = sum(energies) / (num_miners * last_round * runs)
    avg_total_time = sum(total_times) / len(total_times)
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    # --- Throughput (blocks per second) ---
    # each run = one block, total time = avg_total_time * num_miners (approx global wall time)
    # To get throughput, normalize by total experiment time
    T_experiment = avg_total_time / runs
    throughput = runs / T_experiment if T_experiment > 0 else 0
    
    # --- Communication overhead ---
    msgs_per_block = sum(total_msgs) / runs
    bytes_per_block = sum(total_bytes) / runs
    
    # --- Print Summary ---
    print("=== CW-PoW Summary (multi-run) ===")
    #print(f"Runs (blocks simulated): {runs}")
    print(f"Avg total time per miner: {avg_total_time:.2f} s")
    print(f"System throughput: {throughput:.4f} blocks/s")
    print(f"Avg fairness (mean F_i): {avg_fairness:.4f}")
    print(f"Messages per block (total network broadcasts): {msgs_per_block:.2f}")
    print(f"Bytes per block (approx): {bytes_per_block:.1f} bytes")
    print(f"Avg energy per miner (over runs): {avg_energy:.2f} units")
    print(f"Active-Time Fraction (E): {avg_efficiency:.3f}")

    # --- Write CSV ---
    header = [
        "Miner", "Compute_total", "Wait_total", "Hashes_total", "Wins",
        "Msgs_total", "Bytes_total", "P_win", "ResShare", "F_i",
        "Energy", "Total_time"
    ]
    write_csv("cw_pow_results.csv", header, rows)
    
    return {
        "avg_fairness": avg_fairness,
        "avg_total_time": avg_total_time,
        "throughput": throughput,
        "msgs_per_block": msgs_per_block,
        "bytes_per_block": bytes_per_block,
        "avg_energy": avg_energy
    }

# -----------------------
# 3) STANDARD POW (add comm accounting)
# -----------------------
def run_pow(num_miners=NUM_MINERS, blocks_target=1, target_bits=20, seed=42):
    random.seed(seed)
    TARGET = int("388599999999999999999999999999999999999999999999999000000000", 16)#(1 << (256 - target_bits)) - 1
    miners_stats = [{"compute":0.0, "wait":0.0, "wins":0, "hashes":0, "msgs":0, "bytes":0} for _ in range(num_miners)]
    lock = threading.Lock()
    current_block = {"id": 0, "found": False}
    block_count = 0
    start_time = now()

    def miner_thread(i):
        nonlocal block_count
        rng = random.Random(seed + i*10)
        while True:
            with lock:
                if block_count >= blocks_target:
                    break
                blk_id = current_block["id"]
            local_start = now()
            nonce = rng.randint(0, 1 << 3)
            while True:
                h = hashlib.sha256(f"{blk_id}{i}{nonce}".encode()).hexdigest()
                val = int(h, 16)
                miners_stats[i]["hashes"] += 1
                if val <= TARGET:
                    with lock:
                        if blk_id == current_block["id"] and not current_block["found"]:
                            current_block["found"] = True
                            miners_stats[i]["wins"] += 1
                            block_count += 1
                            # broadcast block
                            miners_stats[i]["msgs"] += 1
                            miners_stats[i]["bytes"] += POW_BLOCK_MSG_BYTES
                            current_block["id"] += 1
                            current_block["found"] = False
                    break
                with lock:
                    if blk_id != current_block["id"]:
                        break
                nonce += 1
            local_end = now()
            miners_stats[i]["compute"] += (local_end - local_start)
            miners_stats[i]["wait"] += 0.0

    threads = []
    for i in range(num_miners):
        t = threading.Thread(target=miner_thread, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    total_time = now() - start_time
    rows = []
    resource_shares = []
    for i, s in enumerate(miners_stats):
        resource_shares.append(s["hashes"])
    total_hashes = sum(resource_shares) or 1
    fairness_list = []
    energies = []
    efficiencies = []
    total_msgs = 0
    total_bytes = 0
    for i, s in enumerate(miners_stats):
        share = s["hashes"] / total_hashes
        wins = s["wins"]
        p_win = wins / blocks_target if blocks_target>0 else 0
        F_i = (p_win / share) if share>0 else 0
        fairness_list.append(F_i)
        energy = s["compute"] * POWER_ACTIVE + s["wait"] * POWER_WAIT
        energies.append(energy)
        if (s["compute"] + s["wait"]) > 0:
            E_i = 1 - (s["wait"] / (s["wait"] + s["compute"]))
        else:
            E_i = 0.0
        efficiencies.append(E_i)

        total_msgs += s["msgs"]
        total_bytes += s["bytes"]
        rows.append([i+1, s["compute"], s["wait"], s["hashes"], s["wins"], s["msgs"], s["bytes"], share, p_win, F_i, energy])

    avg_fairness = sum(fairness_list)/len(fairness_list)
    avg_energy = sum(energies)/len(energies)
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    msgs_per_block = total_msgs / blocks_target if blocks_target>0 else 0
    bytes_per_block = total_bytes / blocks_target if blocks_target>0 else 0
    throughput = blocks_target / total_time if total_time>0 else 0

    print("=== PoW Summary ===")
    print(f"Total real time: {total_time:.3f}s, Throughput (blocks/sec): {throughput:.3f}")
    print(f"Avg fairness ratio (mean F_i): {avg_fairness:.3f}")
    print(f"Messages per block (broadcasts): {msgs_per_block:.2f}, Bytes per block: {bytes_per_block:.1f}")
    print(f"Avg energy per miner (over runs): {avg_energy:.2f} units")
    print(f"Active-Time Fraction (E): {avg_efficiency:.3f}")

    write_csv("pow_results.csv", ["Miner","Compute","Wait","Hashes","Wins","Msgs","Bytes","ResShare","P_win","F_i","Energy"], rows)
    return {"time": total_time, "throughput": throughput, "avg_fairness": avg_fairness, "msgs_per_block": msgs_per_block, "bytes_per_block": bytes_per_block}

# -----------------------
# 4) SIMPLE POS (add comm accounting)
# -----------------------
def run_pos(num_validators=50, blocks_target=1, committee_size=11, seed=43):
    random.seed(seed)
    stakes = [random.random() for _ in range(num_validators)]
    total_stake = sum(stakes) or 1
    stakes = [s/total_stake for s in stakes]
    validators_stats = [{"compute":0.0, "wait":0.0, "wins":0, "msgs":0, "bytes":0} for _ in range(num_validators)]
    start_time = now()
    block_count = 0

    for blk in range(blocks_target):
        r = random.random()
        acc = 0.0
        proposer = 0
        for i, s in enumerate(stakes):
            acc += s
            if r <= acc:
                proposer = i
                break
        # proposer broadcasts proposal
        p_start = now()
        time.sleep(0.001)
        p_end = now()
        validators_stats[proposer]["compute"] += (p_end - p_start)
        validators_stats[proposer]["wins"] += 1
        validators_stats[proposer]["msgs"] += 1
        validators_stats[proposer]["bytes"] += POS_PROPOSE_BYTES

        # committee votes
        committee = set()
        while len(committee) < committee_size:
            x = random.randrange(num_validators)
            if x != proposer:
                committee.add(x)
        committee = list(committee)
        for v in committee:
            v_start = now()
            time.sleep(0.002)
            v_end = now()
            validators_stats[v]["compute"] += (v_end - v_start)
            validators_stats[v]["msgs"] += 1
            validators_stats[v]["bytes"] += POS_VOTE_BYTES

        block_count += 1

    total_time = now() - start_time
    rows = []
    fairness_list = []
    energies = []
    efficiencies = []
    total_msgs = 0
    total_bytes = 0
    for i, s in enumerate(validators_stats):
        share = stakes[i]
        wins = s["wins"]
        p_win = wins / blocks_target if blocks_target>0 else 0
        F_i = (p_win / share) if share>0 else 0
        fairness_list.append(F_i)
        energy = s["compute"] * POWER_ACTIVE + s["wait"] * POWER_WAIT
        energies.append(energy)
        if (s["compute"] + s["wait"]) > 0:
            E_i = 1 - (s["wait"] / (s["wait"] + s["compute"]))
        else:
            E_i = 0.0
        efficiencies.append(E_i)
        total_msgs += s["msgs"]
        total_bytes += s["bytes"]
        rows.append([i+1, s["compute"], s["wait"], stakes[i], s["wins"], s["msgs"], s["bytes"], p_win, F_i, energy])

    avg_fairness = sum(fairness_list)/len(fairness_list)
    msgs_per_block = total_msgs / blocks_target if blocks_target>0 else 0
    bytes_per_block = total_bytes / blocks_target if blocks_target>0 else 0
    throughput = blocks_target / total_time if total_time>0 else 0
    avg_energy = sum(energies)/len(energies)
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    
    print("=== PoS Summary ===")
    print(f"Total real time: {total_time:.3f}s, Throughput (blocks/sec): {throughput:.3f}")
    print(f"Avg fairness ratio (mean F_i): {avg_fairness:.3f}")
    print(f"Messages per block: {msgs_per_block:.2f}, Bytes per block: {bytes_per_block:.1f}")
    print(f"Avg energy per miner (over runs): {avg_energy:.2f} units")
    print(f"Active-Time Fraction (E): {avg_efficiency:.3f}")

    write_csv("pos_results.csv", ["Validator","Compute","Wait","Stake","Wins","Msgs","Bytes","P_win","F_i","Energy"], rows)
    return {"time": total_time, "throughput": throughput, "avg_fairness": avg_fairness, "msgs_per_block": msgs_per_block, "bytes_per_block": bytes_per_block}

# -----------------------
# 5) PBFT (add comm accounting)
# -----------------------
def run_pbft(committee_size=13, blocks_target=1, seed=44):
    random.seed(seed)
    assert committee_size >= 4, "committee too small"
    members = [{"compute":0.0, "wait":0.0, "wins":0, "msgs":0, "bytes":0} for _ in range(committee_size)]
    start_time = now()
    block_count = 0

    for blk in range(blocks_target):
        leader = blk % committee_size
        # pre-prepare: leader broadcasts
        lp_start = now()
        time.sleep(0.001)
        lp_end = now()
        members[leader]["compute"] += (lp_end - lp_start)
        members[leader]["msgs"] += 1
        members[leader]["bytes"] += PBFT_PREPARE_BYTES

        # prepare: each member broadcasts prepare
        for i in range(committee_size):
            pv_start = now()
            time.sleep(0.001)
            pv_end = now()
            members[i]["compute"] += (pv_end - pv_start)
            members[i]["msgs"] += 1
            members[i]["bytes"] += PBFT_PREPARE_BYTES

        # commit: each member broadcasts commit
        for i in range(committee_size):
            cv_start = now()
            time.sleep(0.001)
            cv_end = now()
            members[i]["compute"] += (cv_end - cv_start)
            members[i]["msgs"] += 1
            members[i]["bytes"] += PBFT_COMMIT_BYTES

        # finalize when commits >= 2/3
        commits = committee_size  # simulated
        if commits >= ceil(2*committee_size/3):
            block_count += 1
            members[leader]["wins"] += 1

    total_time = now() - start_time
    rows = []
    fairness_list = []
    energies = []
    efficiencies = []
    total_msgs = 0
    total_bytes = 0
    for i, s in enumerate(members):
        share = 1.0 / committee_size
        wins = s["wins"]
        p_win = wins / blocks_target if blocks_target>0 else 0
        F_i = (p_win / share) if share>0 else 0
        fairness_list.append(F_i)
        energy = s["compute"] * POWER_ACTIVE + s["wait"] * POWER_WAIT
        energies.append(energy)
        if (s["compute"] + s["wait"]) > 0:
            E_i = 1 - (s["wait"] / (s["wait"] + s["compute"]))
        else:
            E_i = 0.0
        efficiencies.append(E_i)
        total_msgs += s["msgs"]
        total_bytes += s["bytes"]
        rows.append([i+1, s["compute"], s["wait"], share, s["wins"], s["msgs"], s["bytes"], p_win, F_i, energy])

    avg_fairness = sum(fairness_list)/len(fairness_list)
    msgs_per_block = total_msgs / blocks_target if blocks_target>0 else 0
    bytes_per_block = total_bytes / blocks_target if blocks_target>0 else 0
    throughput = blocks_target / total_time if total_time>0 else 0
    avg_energy = sum(energies)/len(energies)
    avg_efficiency = sum(efficiencies) / len(efficiencies)
    
    print("=== PBFT Summary ===")
    print(f"Total real time: {total_time:.3f}s, Throughput (blocks/sec): {throughput:.3f}")
    print(f"Avg fairness ratio (mean F_i): {avg_fairness:.3f}")
    print(f"Messages per block: {msgs_per_block:.2f}, Bytes per block: {bytes_per_block:.1f}")
    print(f"Avg energy per miner (over runs): {avg_energy:.2f} units")
    print(f"Active-Time Fraction (E): {avg_efficiency:.3f}")

    write_csv("pbft_results.csv", ["Member","Compute","Wait","Share","Wins","Msgs","Bytes","P_win","F_i","Energy"], rows)
    return {"time": total_time, "throughput": throughput, "avg_fairness": avg_fairness, "msgs_per_block": msgs_per_block, "bytes_per_block": bytes_per_block}

# -----------------------
# 6) ABLATION / ROBUSTNESS EXPERIMENTS (added for revision)
# -----------------------
def run_ablation_apt(num_miners=12, runs=8, seed=2025):
    """
    Sensitivity/ablation study over the initial APT operating point (NbrR, NbrS).
    Each configuration is run independently (APT still adapts (NbrR, NbrS) from
    that starting point every block, as in the main experiment).
    """
    configs = [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6),
               (6, 1), (6, 6), (2, 6)]
    rows = []
    for (r0, s0) in configs:
        stats = run_enhanced_cw_pow(num_miners=num_miners, nbrR=r0, nbrS=s0,
                                     runs=runs, seed=seed)
        rows.append([r0, s0, stats["final_params"][0], stats["final_params"][1],
                     stats["throughput"], stats["avg_total_time"],
                     stats["avg_fairness"], stats["avg_energy_per_round"],
                     stats["msgs_per_block"]])
        print(f"(NbrR0={r0:>2}, NbrS0={s0:>2}) -> final=({stats['final_params'][0]},{stats['final_params'][1]}) "
              f"Th={stats['throughput']:.3f} TT={stats['avg_total_time']:.3f} "
              f"F={stats['avg_fairness']:.3f} AEM/round={stats['avg_energy_per_round']:.3f} "
              f"MB={stats['msgs_per_block']:.1f}")
    write_csv("ablation_apt_nbrr_nbrs.csv",
              ["NbrR0", "NbrS0", "NbrR_final", "NbrS_final", "Throughput",
               "AvgTotalTime", "AvgFairness", "AvgEnergyPerRound", "MsgsPerBlock"],
              rows)
    return rows


def run_byzantine_robustness(num_miners=12, runs=8, seed=2025):
    """
    Robustness of the commit-reveal round identifier and overall liveness when
    a fraction of miners commit but withhold their reveal (Byzantine
    solution-withholding), with a bounded reveal-or-forfeit timeout so the
    protocol still makes progress.
    """
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for frac in fractions:
        stats = run_enhanced_cw_pow(num_miners=num_miners, runs=runs, seed=seed,
                                     byzantine_withhold_frac=frac,
                                     reveal_timeout_polls=500)
        rows.append([frac, stats["throughput"], stats["avg_total_time"],
                     stats["avg_fairness"], stats["avg_energy_per_round"]])
        print(f"withhold_frac={frac:.1f} -> Th={stats['throughput']:.3f} "
              f"TT={stats['avg_total_time']:.3f} F={stats['avg_fairness']:.3f} "
              f"AEM/round={stats['avg_energy_per_round']:.3f}")
    write_csv("byzantine_withholding_robustness.csv",
              ["WithholdFraction", "Throughput", "AvgTotalTime", "AvgFairness",
               "AvgEnergyPerRound"], rows)
    return rows


def run_network_spike_robustness(num_miners=12, runs=8, seed=2025):
    """
    Impact of bursty network conditions (random propagation delay spikes
    affecting round-solution broadcasts) on latency and throughput.
    """
    settings = [(0.0, 0.0), (0.1, 0.05), (0.3, 0.05), (0.3, 0.15)]
    rows = []
    for (prob, dur) in settings:
        stats = run_enhanced_cw_pow(num_miners=num_miners, runs=runs, seed=seed,
                                     network_delay_spike_prob=prob,
                                     network_delay_spike_sec=dur,
                                     reveal_timeout_polls=500)
        rows.append([prob, dur, stats["throughput"], stats["avg_total_time"],
                     stats["avg_fairness"]])
        print(f"spike_prob={prob:.1f} spike_sec={dur:.2f} -> "
              f"Th={stats['throughput']:.3f} TT={stats['avg_total_time']:.3f} "
              f"F={stats['avg_fairness']:.3f}")
    write_csv("network_delay_spike_robustness.csv",
              ["SpikeProb", "SpikeSec", "Throughput", "AvgTotalTime", "AvgFairness"],
              rows)
    return rows


# -----------------------
# MAIN: run ECW-PoW + CW-PoW + PoW + PoS + PBFT
# -----------------------
if __name__ == "__main__":
    
    print("Running enhanced CW-PoW ...")
    stats = run_enhanced_cw_pow()
    print()
    
    print("Running CW-PoW ...")
    cw_stats = run_cw_pow(num_miners=NUM_MINERS, last_round=5, sol_nb=3, runs=RUNS, target_bits=24, seed=1000)
    print()

    print("Running standard PoW...")
    pow_stats = run_pow(num_miners=NUM_MINERS, blocks_target=1, target_bits=20, seed=2000)
    print()

    print("Running PoS...")
    pos_stats = run_pos(num_validators=50, blocks_target=50, committee_size=11, seed=3000)
    print()

    print("Running PBFT...")
    pbft_stats = run_pbft(committee_size=13, blocks_target=40, seed=4000)
    print()

    print("All simulations completed. CSV files generated for each algorithm.")
    print()

    print("Running APT initial-parameter ablation study ...")
    run_ablation_apt()
    print()

    print("Running Byzantine withholding robustness study ...")
    run_byzantine_robustness()
    print()

    print("Running network delay-spike robustness study ...")
    run_network_spike_robustness()
    print()

    print("All robustness/ablation studies completed. CSV files generated.")


