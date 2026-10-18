"""
extended_baseline_comparison.py

Runs classical PoW, our simplified reimplementation of Green-PoW (Lasla et
al., 2022), and our simplified reimplementation of Proof-of-Experience
(Masseport et al., 2023) under IDENTICAL conditions (same miner count,
block count, difficulty target, timing/energy model) within this paper's
simulator, to produce a normalized, like-for-like comparison -- addressing
the reviewer request for a quantitative comparison against modern
low-energy permissionless PoW baselines.
"""
import importlib.util
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "EnhancedCW 5 algo.py")
spec = importlib.util.spec_from_file_location("enhanced_cw_algo", MODULE_PATH)
algo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algo)

NUM_MINERS = 10
BLOCKS = 20

print("### Classical PoW ###")
pow_stats = algo.run_pow(num_miners=NUM_MINERS, blocks_target=BLOCKS, target_bits=20, seed=2000)
print()

print("### Green-PoW (simplified reimplementation) ###")
green_stats = algo.run_green_pow(num_miners=NUM_MINERS, blocks_target=BLOCKS, seed=5000)
print()

print("### Proof-of-Experience (simplified reimplementation) ###")
poe_stats = algo.run_proof_of_experience(num_miners=NUM_MINERS, blocks_target=BLOCKS, seed=6000)
print()

rows = [
    ["PoW", pow_stats["time"], pow_stats["throughput"], pow_stats["avg_fairness"],
     pow_stats["msgs_per_block"], pow_stats["bytes_per_block"], pow_stats["avg_energy"]],
    ["Green-PoW", green_stats["time"], green_stats["throughput"], green_stats["avg_fairness"],
     green_stats["msgs_per_block"], green_stats["bytes_per_block"], green_stats["avg_energy"]],
    ["Proof-of-Experience", poe_stats["time"], poe_stats["throughput"], poe_stats["avg_fairness"],
     poe_stats["msgs_per_block"], poe_stats["bytes_per_block"], poe_stats["avg_energy"]],
]

print("=== Summary table (num_miners=10, blocks=20) ===")
print(f"{'Algorithm':<22}{'TT(s)':<10}{'Th':<10}{'F':<10}{'MB':<8}{'BB':<12}{'AEM':<10}")
for r in rows:
    print(f"{r[0]:<22}{r[1]:<10.3f}{r[2]:<10.4f}{r[3]:<10.4f}{r[4]:<8.1f}{r[5]:<12.1f}{r[6]:<10.2f}")

csv_path = os.path.join(HERE, "extended_baseline_comparison.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Algorithm", "TT_s", "Th_blocks_per_s", "F", "MB", "BB_bytes", "AEM"])
    w.writerows(rows)
print(f"\nWrote {csv_path}")
