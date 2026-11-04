"""SolarGuard AI — k-NN Curve Memory Model.

Step 1: Build Clean Memory Bank
Step 2: Compute Leave-One-Out Distance Distribution & Propose Cutoff
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=== ENVIRONMENT & LIBRARY VERSIONS ===")
print("python    :", sys.version.split()[0])
print("pandas    :", pd.__version__)
print("numpy     :", np.__version__)
print("matplotlib:", matplotlib.__version__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN_CSV = os.path.join(ROOT, "outputs", "clean_table.csv")
RANK_CSV = os.path.join(ROOT, "outputs", "final_daily_ranking.csv")

# ---------------- Step 1: Load & Prepare Data ----------------
print("\n=== STEP 1: LOAD & PREPARE CURVE MEMORY BANK ===")
df_clean = pd.read_csv(CLEAN_CSV)
df_rank = pd.read_csv(RANK_CSV)
print(f"clean_table rows: {len(df_clean)}, cols: {list(df_clean.columns)}")
print(f"final_daily_ranking rows: {len(df_rank)}, cols: {len(df_rank.columns)}")

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean["time_of_day"] = df_clean["timestamp"].dt.strftime("%H:%M")

# Pivot across all 51 time-of-day slots (06:00 to 18:30)
pivot_curves = df_clean.pivot_table(
    index=["date", "SOURCE_KEY"],
    columns="time_of_day",
    values="AC_POWER",
).fillna(0.0)

all_idx = df_rank[["date", "SOURCE_KEY"]].set_index(["date", "SOURCE_KEY"]).index
all_curves = pivot_curves.reindex(all_idx).to_numpy()

clean_mask = (df_rank["flagged"] == "no").to_numpy()
memory_bank = all_curves[clean_mask]

print(f"Total inverter-days in dataset: {len(all_curves)}")
print(f"Clean unflagged inverter-days in Memory Bank: {len(memory_bank)}")
print(f"Flagged inverter-days excluded from Memory Bank: {(~clean_mask).sum()}")
print(f"Memory Bank matrix shape: {memory_bank.shape} (profiles x daytime slots)")

# ---------------- Step 2: Compute Distance Distribution ----------------
print("\n=== STEP 2: LEAVE-ONE-OUT DISTANCE DISTRIBUTION (CLEAN MEMORY) ===")
# Pairwise Euclidean distance within clean memory
dot_mem = np.dot(memory_bank, memory_bank.T)
sq_mem = np.sum(memory_bank**2, axis=1)
dists_sq = np.maximum(0, sq_mem[:, None] + sq_mem[None, :] - 2 * dot_mem)
dists = np.sqrt(dists_sq)

# Leave-one-out: exclude distance to self
np.fill_diagonal(dists, np.inf)
sorted_dists = np.sort(dists, axis=1)

for k_val in [1, 3, 5]:
    k_mean = np.mean(sorted_dists[:, :k_val], axis=1)
    s = pd.Series(k_mean)
    print(f"\nDistance distribution to k={k_val} nearest neighbours in clean memory:")
    print(f"  min     : {s.min():.6f}")
    print(f"  25%     : {s.quantile(0.25):.6f}")
    print(f"  median  : {s.median():.6f}")
    print(f"  75%     : {s.quantile(0.75):.6f}")
    print(f"  95%     : {s.quantile(0.95):.6f}")
    print(f"  99%     : {s.quantile(0.99):.6f}")
    print(f"  max     : {s.max():.6f}")
