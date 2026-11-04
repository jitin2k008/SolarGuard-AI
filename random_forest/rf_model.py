"""SolarGuard AI — Option A: Random Forest Regressor Model.

Trains a Random Forest Regressor on solar generation data:
- Features: Diurnal time cycle (minute of day, sin/cos), live spatial peer features (peer median, Q25, Q75, max).
- Training: Clean unflagged inverter-days on Days 1-24.
- Testing: Days 25-34.
- Evaluation: R2 score, RMSE, MAE.
- Anomaly Detection: Daily RF gap % and MAD scaled deviation (k=4) across all 748 inverter-days.
- Multi-Model Consensus: 3-way agreement (Peer Baseline, k-NN Memory, Random Forest).
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

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

t_start = time.perf_counter()

print("=== ENVIRONMENT & LIBRARY VERSIONS ===")
print("python      :", sys.version.split()[0])
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("matplotlib  :", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLEAN_CSV = os.path.join(ROOT, "outputs", "clean_table.csv")
RANK_CSV = os.path.join(ROOT, "outputs", "final_daily_ranking.csv")
KNN_CSV = os.path.join(ROOT, "memoryNN", "knn_memory_ranking.csv")
OUT_CSV = os.path.join(HERE, "rf_model_ranking.csv")
OUT_PNG = os.path.join(HERE, "plot_rf_prediction_example.png")

# ---------------- Step 1: Load Data & Feature Engineering ----------------
print("\n=== STEP 1: LOAD & FEATURE ENGINEERING ===")
df_clean = pd.read_csv(CLEAN_CSV)
df_rank = pd.read_csv(RANK_CSV)
df_knn = pd.read_csv(KNN_CSV)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])
df_clean["date"] = pd.to_datetime(df_clean["date"]).dt.strftime("%Y-%m-%d")
df_rank["date"] = pd.to_datetime(df_rank["date"]).dt.strftime("%Y-%m-%d")
df_knn["date"] = pd.to_datetime(df_knn["date"]).dt.strftime("%Y-%m-%d")

# Peer statistics per timestamp
piv_ac = df_clean.pivot_table(index="timestamp", columns="SOURCE_KEY", values="AC_POWER")
vals = piv_ac.to_numpy(dtype=float)
n_inv = vals.shape[1]

peer_med = np.full(vals.shape, np.nan)
peer_q25 = np.full(vals.shape, np.nan)
peer_q75 = np.full(vals.shape, np.nan)
peer_max = np.full(vals.shape, np.nan)

for j in range(n_inv):
    others = np.delete(vals, j, axis=1)
    peer_med[:, j] = np.nanmedian(others, axis=1)
    peer_q25[:, j] = np.nanpercentile(others, 25, axis=1)
    peer_q75[:, j] = np.nanpercentile(others, 75, axis=1)
    peer_max[:, j] = np.nanmax(others, axis=1)

ts_pos = {ts: i for i, ts in enumerate(piv_ac.index)}
col_pos = {k: j for j, k in enumerate(piv_ac.columns)}
ti = df_clean["timestamp"].map(ts_pos).to_numpy()
sj = df_clean["SOURCE_KEY"].map(col_pos).to_numpy()

df_clean["peer_median_ac"] = peer_med[ti, sj]
df_clean["peer_q25_ac"] = peer_q25[ti, sj]
df_clean["peer_q75_ac"] = peer_q75[ti, sj]
df_clean["peer_max_ac"] = peer_max[ti, sj]

# Diurnal time cycle features
minute_of_day = df_clean["timestamp"].dt.hour * 60 + df_clean["timestamp"].dt.minute
df_clean["minute_of_day"] = minute_of_day
df_clean["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440)
df_clean["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440)
df_clean["day_of_year"] = df_clean["timestamp"].dt.day_of_year

# Filter valid rows (where peer median is valid)
valid_mask = ~df_clean["peer_median_ac"].isna()
df_model = df_clean[valid_mask].copy()

# Merge clean flag to select training rows
df_model = df_model.merge(df_rank[["date", "SOURCE_KEY", "flagged"]], on=["date", "SOURCE_KEY"], how="left")

feature_cols = ["minute_of_day", "sin_time", "cos_time", "peer_median_ac", "peer_q25_ac", "peer_q75_ac", "peer_max_ac", "day_of_year"]
target_col = "AC_POWER"

print(f"Total modeling samples: {len(df_model)}")
print(f"Features: {feature_cols}")

# ---------------- Step 2: Time-Split Train & Test ----------------
unique_dates = sorted(df_model["date"].unique())
split_date = unique_dates[24]  # Days 1-24 train, days 25-34 test
print(f"Train period: {unique_dates[0]} to {unique_dates[23]} (24 days)")
print(f"Test period : {split_date} to {unique_dates[-1]} (10 days)")

train_mask = (df_model["date"] < split_date) & (df_model["flagged"] == "no")
test_mask = df_model["date"] >= split_date

X_train = df_model.loc[train_mask, feature_cols]
y_train = df_model.loc[train_mask, target_col]

X_test = df_model.loc[test_mask, feature_cols]
y_test = df_model.loc[test_mask, target_col]

print(f"Clean training samples: {len(X_train)}")
print(f"Test samples (all inverters): {len(X_test)}")

# ---------------- Step 3: Train Random Forest Regressor ----------------
print("\n=== STEP 2: TRAINING RANDOM FOREST REGRESSOR ===")
t_train = time.perf_counter()
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
train_time = time.perf_counter() - t_train
print(f"Model trained in {train_time:.4f} seconds.")

# Performance metrics
train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

r2_train = r2_score(y_train, train_pred)
r2_test = r2_score(y_test, test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))
mae_test = mean_absolute_error(y_test, test_pred)

print(f"Training R2 Score: {r2_train * 100:.2f}%")
print(f"Test R2 Score    : {r2_test * 100:.2f}%")
print(f"Test RMSE        : {rmse_test:.4f}")
print(f"Test MAE         : {mae_test:.4f}")

# Feature importances
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
for f, v in imp.items():
    print(f"  {f:18s}: {v * 100:.2f}%")

# ---------------- Step 4: Full Dataset Inference & Anomaly Detection ----------------
print("\n=== STEP 3: FULL DATASET INFERENCE & ANOMALY DETECTION ===")
df_model["rf_predicted"] = rf.predict(df_model[feature_cols])

# Daily aggregation per inverter-day
daily_rf = (
    df_model.groupby(["date", "SOURCE_KEY"])
    .agg(
        rf_actual_sum=("AC_POWER", "sum"),
        rf_predicted_sum=("rf_predicted", "sum"),
        coverage=("AC_POWER", "size"),
    )
    .reset_index()
)
daily_rf["rf_gap_pct"] = (daily_rf["rf_predicted_sum"] - daily_rf["rf_actual_sum"]) / daily_rf["rf_predicted_sum"] * 100

# MAD-based outlier detection across each day
day_med = daily_rf.groupby("date")["rf_gap_pct"].transform("median")
day_mad = daily_rf.groupby("date")["rf_gap_pct"].transform(lambda x: np.median(np.abs(x - np.median(x))))
denom = 1.4826 * day_mad
daily_rf["rf_scaled_dev"] = np.where(denom > 0, (daily_rf["rf_gap_pct"] - day_med) / denom, 0.0)
daily_rf["rf_flagged"] = np.where(daily_rf["rf_scaled_dev"] > 4.0, "yes", "no")

print(f"Total inverter-days scored: {len(daily_rf)}")
print(f"Random Forest flags count: {dict(daily_rf['rf_flagged'].value_counts())}")

# ---------------- Step 5: Three-Way Consensus Comparison ----------------
print("\n=== STEP 4: THREE-WAY MODEL CONSENSUS ===")
# Merge Base and k-NN flags
comp = daily_rf[["date", "SOURCE_KEY", "rf_gap_pct", "rf_scaled_dev", "rf_flagged"]].copy()
comp = comp.merge(df_rank[["date", "SOURCE_KEY", "rank", "gap_pct", "scaled_dev", "flagged"]].rename(
    columns={"flagged": "base_flagged", "gap_pct": "base_gap_pct", "scaled_dev": "base_scaled_dev"}), on=["date", "SOURCE_KEY"])
comp = comp.merge(df_knn[["date", "SOURCE_KEY", "knn_distance_score", "knn_flagged"]], on=["date", "SOURCE_KEY"])

triple_flagged = (comp["base_flagged"] == "yes") & (comp["knn_flagged"] == "yes") & (comp["rf_flagged"] == "yes")
triple_normal = (comp["base_flagged"] == "no") & (comp["knn_flagged"] == "no") & (comp["rf_flagged"] == "no")

print(f"Triple-Confirmed Flagged (All 3 Models Agree): {triple_flagged.sum()} days")
print(f"Triple-Confirmed Normal  (All 3 Models Agree): {triple_normal.sum()} days")
print(f"Total 3-Model Consensus Agreement: {(triple_flagged.sum() + triple_normal.sum()) / len(comp) * 100:.2f}%")

# ---------------- Step 6: Save Output & Plot ----------------
comp.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV} ({os.path.getsize(OUT_CSV)} bytes, {len(comp)} rows)")

with open(OUT_CSV, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest().upper()
print(f"SHA256: {sha}")

# Example Plot: bvBOhCH3iADSZry on 2020-06-14
target_date = "2020-06-14"
target_inv = "bvBOhCH3iADSZry"

one = df_model[(df_model["date"] == target_date) & (df_model["SOURCE_KEY"] == target_inv)].sort_values("timestamp")

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(one["timestamp"], one["rf_predicted"], color="#1f77b4", linewidth=2, label="Random Forest Predicted AC_POWER")
ax.plot(one["timestamp"], one["AC_POWER"], color="#d62728", linewidth=2, label="Actual AC_POWER")
ax.fill_between(
    one["timestamp"], one["AC_POWER"], one["rf_predicted"],
    where=one["rf_predicted"] >= one["AC_POWER"],
    color="#d62728", alpha=0.15, label="Predicted Shortfall"
)
ax.set_title(f"Random Forest Prediction: {target_inv} on 2020-06-14 (R2={r2_test*100:.1f}%)", fontsize=11, fontweight="bold")
ax.set_xlabel("Timestamp")
ax.set_ylabel("AC_POWER (as given in the file)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
plt.close(fig)

print(f"Saved plot: {OUT_PNG} ({os.path.getsize(OUT_PNG)} bytes)")
print(f"Total script run time: {time.perf_counter() - t_start:.4f} seconds")
