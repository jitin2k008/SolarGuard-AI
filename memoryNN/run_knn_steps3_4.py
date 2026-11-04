"""SolarGuard AI — k-NN Curve Memory Model.

Step 3: Anomaly Scoring & Evaluation (k=3, cutoff=398.458949)
Step 4: Save Tables, Generate Plot & Compute Checksum
"""

import hashlib
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_start = time.perf_counter()

print("=== ENVIRONMENT & LIBRARY VERSIONS ===")
print("python    :", sys.version.split()[0])
print("pandas    :", pd.__version__)
print("numpy     :", np.__version__)
print("matplotlib:", matplotlib.__version__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN_CSV = os.path.join(ROOT, "outputs", "clean_table.csv")
RANK_CSV = os.path.join(ROOT, "outputs", "final_daily_ranking.csv")
OUT_CSV = os.path.join(HERE, "knn_memory_ranking.csv")
OUT_PNG = os.path.join(HERE, "plot_knn_memory_example.png")

# Approved parameters
K_VAL = 3
CUTOFF_95 = 398.458949

# ---------------- Step 3: Anomaly Scoring ----------------
df_clean = pd.read_csv(CLEAN_CSV)
df_rank = pd.read_csv(RANK_CSV)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean["time_of_day"] = df_clean["timestamp"].dt.strftime("%H:%M")
time_slots = sorted(df_clean["time_of_day"].unique())

pivot_curves = df_clean.pivot_table(
    index=["date", "SOURCE_KEY"],
    columns="time_of_day",
    values="AC_POWER",
).fillna(0.0)

all_idx = df_rank[["date", "SOURCE_KEY"]].set_index(["date", "SOURCE_KEY"]).index
all_curves = pivot_curves.reindex(all_idx).to_numpy()

clean_mask = (df_rank["flagged"] == "no").to_numpy()
memory_bank = all_curves[clean_mask]
memory_index = df_rank[clean_mask][["date", "SOURCE_KEY"]].reset_index(drop=True)

# Pairwise distances from all 748 curves to 696 clean memory curves
dot = np.dot(all_curves, memory_bank.T)
sq_all = np.sum(all_curves**2, axis=1)
sq_mem = np.sum(memory_bank**2, axis=1)
dists_sq = np.maximum(0, sq_all[:, None] + sq_mem[None, :] - 2 * dot)
dists = np.sqrt(dists_sq)

# Leave-one-out for clean curves matching memory bank
clean_indices = np.where(clean_mask)[0]
for i_mem, i_all in enumerate(clean_indices):
    dists[i_all, i_mem] = np.inf

# Find k nearest neighbors
sorted_order = np.argsort(dists, axis=1)
sorted_dists = np.take_along_axis(dists, sorted_order, axis=1)

k3_scores = np.mean(sorted_dists[:, :K_VAL], axis=1)
knn_flags = np.where(k3_scores > CUTOFF_95, "yes", "no")

t_scoring = time.perf_counter() - t_start

# Build results table
res = df_rank[["date", "SOURCE_KEY", "rank", "gap_pct", "scaled_dev", "flagged"]].copy()
res["knn_distance_score"] = np.round(k3_scores, 6)
res["knn_flagged"] = knn_flags
res["nn_1_dist"] = np.round(sorted_dists[:, 0], 6)
res["nn_2_dist"] = np.round(sorted_dists[:, 1], 6)
res["nn_3_dist"] = np.round(sorted_dists[:, 2], 6)

print(f"\n=== STEP 3: EVALUATION & TIMING ===")
print(f"Memory query execution time: {t_scoring:.4f} seconds")
print(f"Total inverter-days scored: {len(res)}")
print(f"k-NN Model flags count: {dict(res['knn_flagged'].value_counts())}")

# 2x2 Contingency Table
ct = pd.crosstab(res["flagged"], res["knn_flagged"], rownames=["Peer Baseline (flagged)"], colnames=["Memory Model (knn_flagged)"])
print("\n=== CONTINGENCY TABLE (Peer vs Memory Model) ===")
print(ct)

both_yes = ((res["flagged"] == "yes") & (res["knn_flagged"] == "yes")).sum()
both_no = ((res["flagged"] == "no") & (res["knn_flagged"] == "no")).sum()
peer_only = ((res["flagged"] == "yes") & (res["knn_flagged"] == "no")).sum()
knn_only = ((res["flagged"] == "no") & (res["knn_flagged"] == "yes")).sum()

print(f"\nBreakdown:")
print(f"  Agreed Flagged (both yes): {both_yes}")
print(f"  Agreed Normal  (both no) : {both_no}")
print(f"  Peer Only      (peer yes, knn no): {peer_only}")
print(f"  Memory Only    (peer no, knn yes): {knn_only}")

# ---------------- Step 4: Save & Visualize ----------------
res.to_csv(OUT_CSV, index=False)
print(f"\n=== STEP 4: OUTPUT SAVED ===")
print(f"Saved: {OUT_CSV} ({os.path.getsize(OUT_CSV)} bytes, {len(res)} rows)")

# SHA256
with open(OUT_CSV, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest().upper()
print(f"SHA256: {sha}")

# Generate Example Plot
target_date = "2020-06-14"
target_inv = "bvBOhCH3iADSZry"
row_match = res[(res["date"] == target_date) & (res["SOURCE_KEY"] == target_inv)]
target_idx = row_match.index[0]

target_curve = all_curves[target_idx]
target_score = k3_scores[target_idx]
target_nn_idx = sorted_order[target_idx, :K_VAL]

fig, ax = plt.subplots(figsize=(10, 4.8))
x_ticks = range(len(time_slots))

# Plot the 3 nearest historical curves retrieved from memory
colors = ["#2ca02c", "#17becf", "#9467bd"]
for rank_i, (mem_i, col) in enumerate(zip(target_nn_idx, colors)):
    nn_date = memory_index.loc[mem_i, "date"]
    nn_inv = memory_index.loc[mem_i, "SOURCE_KEY"]
    nn_dist = sorted_dists[target_idx, rank_i]
    ax.plot(x_ticks, memory_bank[mem_i], color=col, linestyle="--", alpha=0.85,
            label=f"Memory #{rank_i+1}: {nn_inv} ({nn_date}), dist={nn_dist:.2f}")

# Plot target curve
ax.plot(x_ticks, target_curve, color="#d62728", linewidth=2.5,
        label=f"Tested: {target_inv} ({target_date}), score={target_score:.2f}")

ax.axhline(0, color="gray", linewidth=0.5)
ax.set_title(f"k-NN Memory Model: {target_inv} on {target_date} vs 3 Nearest Memory Curves", fontsize=11, fontweight="bold")
ax.set_xlabel("Time of Day")
ax.set_ylabel("AC_POWER (as given in the file)")
ax.set_xticks(x_ticks[::4])
ax.set_xticklabels(time_slots[::4], rotation=45, fontsize=8)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)

print(f"Saved plot: {OUT_PNG} ({os.path.getsize(OUT_PNG)} bytes)")
print(f"Total script run time: {time.perf_counter() - t_start:.4f} seconds")
