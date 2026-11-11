"""
proto_benchmarks.py
===================
Host dependence of the multi-threaded prototype ("EnhancedCW 5 algo.py"), quoted in Sections 5.2 and 5.5 of the
manuscript.

  1. per-hash wall-clock time of one thread running alone versus ten threads running together
     (interpreter-lock contention);
  2. per-block values (wall-clock time per block, energy per miner and block) of PoW, CW-PoW, and Enhanced CW-PoW measured by the prototype in the reference
     configurations (10 miners, 20 blocks, three repetitions), for comparison with the event-driven model.

    python proto_benchmarks.py            ->  prototype_crosscheck.csv, prototype_benchmarks.txt

The numbers depend on the host. They are NOT used for conclusions in the manuscript.
"""
import contextlib
import csv
import hashlib
import importlib.util
import io
import os
import statistics
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("proto", os.path.join(HERE, "EnhancedCW 5 algo.py"))
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)


def hash_loop(n, x=123456789):
    t0 = time.perf_counter()
    for nonce in range(1, n + 1):
        int(hashlib.sha256(f"{nonce}0{x}".encode()).hexdigest(), 16)
    return (time.perf_counter() - t0) / n


def contention(threads=10, n=250000, reps=2):
    alone = statistics.mean(hash_loop(n // 2) for _ in range(reps))
    ratios = []
    for _ in range(reps):
        res = [0.0] * threads

        def work(i):
            res[i] = hash_loop(n)

        ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ratios.append(statistics.mean(res) / alone)
    return alone, statistics.mean(ratios)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **k)


def main(reps=3, blocks=20):
    lines = []
    alone, ratio = contention()
    lines.append(f"one hash attempt, one thread alone: {alone * 1e6:.2f} us")
    lines.append(f"one hash attempt, 10 threads together: {alone * ratio * 1e6:.2f} us  (x{ratio:.1f})")
    rows = []
    contended = []
    for k in range(reps):
        p = quiet(proto.run_pow, num_miners=10, blocks_target=blocks, seed=2000 + k)
        t0 = time.perf_counter()
        c = quiet(proto.run_cw_pow, num_miners=10, last_round=5, sol_nb=3, runs=blocks, seed=1000 + k)
        tc = (time.perf_counter() - t0) / blocks            # wall-clock time per block
        t0 = time.perf_counter()
        e = quiet(proto.run_enhanced_cw_pow, num_miners=10, runs=blocks, seed=2025 + k)
        te = (time.perf_counter() - t0) / blocks
        with open(os.path.join(HERE, "pow_results.csv"), newline="") as fh:       # written by run_pow()
            pr = list(csv.DictReader(fh))
        contended.append(sum(float(r["Compute"]) for r in pr) / sum(float(r["Hashes"]) for r in pr))
        rows.append(["PoW", k, p["time"] / blocks, p["avg_energy"] / blocks, p["msgs_per_block"], p["bytes_per_block"]])
        rows.append(["CW-PoW", k, tc, c["avg_energy_per_block"], c["msgs_per_block"], c["bytes_per_block"]])
        rows.append(["Enhanced CW-PoW", k, te, e["avg_energy_per_block"], e["msgs_per_block"], e["bytes_per_block"]])
    lines.append(f"PoW run, ten threads: wall-clock time per hash and thread {statistics.mean(contended) * 1e6:.2f} us")
    with open(os.path.join(HERE, "prototype_crosscheck.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "rep", "time_per_block_s", "energy_per_miner_block", "msgs_per_block", "bytes_per_block"])
        w.writerows(rows)
    for name in ("PoW", "CW-PoW", "Enhanced CW-PoW"):
        rr = [r for r in rows if r[0] == name]
        lines.append(f"{name:16s} time/block {statistics.mean(r[2] for r in rr) * 1e3:8.1f} ms   energy/block "
                     f"{statistics.mean(r[3] for r in rr):7.2f}   msgs {statistics.mean(r[4] for r in rr):6.1f}   "
                     f"bytes {statistics.mean(r[5] for r in rr):10.0f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "prototype_benchmarks.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
