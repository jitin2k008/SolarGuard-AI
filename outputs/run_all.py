"""Rebuild every SolarGuard output from the original CSV, using the approved values only.

Run:  python outputs/run_all.py

Reads : Plant_1_Generation_Data.csv.zip   (never modified)
Writes: outputs/clean_table.csv, expected_table.csv, daily_ranking.csv,
        cause_hypotheses.csv, final_daily_ranking.csv,
        plot_a_actual_vs_expected.png, plot_b_gap_grid.png

Approved values are the constants below. They are the values recorded in
PROJECT_STATE.md and must not be changed without the user typing APPROVED.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP = os.path.join(ROOT, "Plant_1_Generation_Data.csv.zip")
OUT = HERE

# ---- approved values, do not edit without approval -------------------------
MIN_PEERS = 10          # P3
K_FLAG = 4.0            # P4
LOW_COVERAGE_SHARE = 95.0   # P4
TIE_RULE = "gap_pct descending, ties broken by SOURCE_KEY ascending"
THR_H1 = 0.80           # P5
THR_H2 = 0.85
THR_H3 = 2
THR_H4 = 0.10
THR_H5 = 0.05
SPLIT_INDEX = 24        # P6, days 1-24 derive, 25-34 apply
PLOT_A_INVERTER = "bvBOhCH3iADSZry"
PLOT_A_DATE = "2020-06-14"


def log(msg):
    print(msg, flush=True)


def main():
    t_start = time.perf_counter()
    log("python  " + sys.version.split()[0])
    log("pandas  " + pd.__version__)
    log("numpy   " + np.__version__)
    log("matplotlib " + matplotlib.__version__)
    log("input   " + ZIP)
    log("output  " + OUT)
    log("")

    # ---------------- P1: read and parse ----------------
    t = time.perf_counter()
    df = pd.read_csv(ZIP, compression="zip")
    df["timestamp"] = pd.to_datetime(df["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["date"] = df["timestamp"].dt.normalize().dt.date
    log(f"P1 rows read: {len(df)}  parse failures: {int(df['timestamp'].isna().sum())}  "
        f"({time.perf_counter() - t:.4f} s)")

    # ---------------- P2: clean table ----------------
    t = time.perf_counter()
    pivot_day = df.pivot_table(index="timestamp", columns="SOURCE_KEY", values="AC_POWER")
    n_present = pivot_day.notna().sum(axis=1)
    is_day = (pivot_day > 0).sum(axis=1) > (n_present / 2)
    day_set = set(is_day[is_day].index)
    table = df[df["timestamp"].isin(day_set)][
        ["timestamp", "date", "SOURCE_KEY", "DC_POWER", "AC_POWER"]].copy()
    table["was_filled"] = False
    table["was_missing"] = False
    rows_in = len(df)
    table.to_csv(os.path.join(OUT, "clean_table.csv"), index=False)
    log(f"P2 daytime timestamps: {len(day_set)}  rows {rows_in} -> {len(table)}  "
        f"({time.perf_counter() - t:.4f} s)")

    # ---------------- P3: expected from peers ----------------
    t = time.perf_counter()
    piv_ac = table.pivot_table(index="timestamp", columns="SOURCE_KEY", values="AC_POWER")
    vals = piv_ac.to_numpy(dtype=float)
    n_pr = np.sum(~np.isnan(vals), axis=1)
    expected = np.full(vals.shape, np.nan)
    for i in range(vals.shape[0]):
        if n_pr[i] - 1 < MIN_PEERS:
            continue
        row = vals[i]
        for j in range(vals.shape[1]):
            if np.isnan(row[j]):
                continue
            others = np.delete(row, j)
            others = others[~np.isnan(others)]
            expected[i, j] = np.median(others)

    ts_pos = {ts: i for i, ts in enumerate(piv_ac.index)}
    col_pos = {k: j for j, k in enumerate(piv_ac.columns)}
    ti = table["timestamp"].map(ts_pos).to_numpy()
    sj = table["SOURCE_KEY"].map(col_pos).to_numpy()

    exp_rows = table[["timestamp", "date", "SOURCE_KEY", "AC_POWER"]].copy()
    exp_rows["expected"] = expected[ti, sj]
    exp_rows["n_peers"] = (n_pr - 1)[ti]
    exp_rows["expected_valid"] = ~exp_rows["expected"].isna()
    exp_rows.to_csv(os.path.join(OUT, "expected_table.csv"), index=False)
    log(f"P3 expected values: {int(exp_rows['expected_valid'].sum())} of {len(exp_rows)}  "
        f"({time.perf_counter() - t:.4f} s)")

    # ---------------- P4: gap_pct, rank, flags ----------------
    t = time.perf_counter()
    u = exp_rows[exp_rows["expected_valid"]].copy()
    g = (u.groupby(["SOURCE_KEY", "date"])
         .agg(actual_sum=("AC_POWER", "sum"),
              expected_sum=("expected", "sum"),
              coverage=("AC_POWER", "size"))
         .reset_index())
    g["gap_pct"] = (g["expected_sum"] - g["actual_sum"]) / g["expected_sum"] * 100

    day_ts = u.groupby("date")["timestamp"].nunique()
    g["day_timestamps"] = g["date"].map(day_ts)
    g["coverage_share"] = g["coverage"] / g["day_timestamps"] * 100

    day_med = g.groupby("date")["gap_pct"].transform("median")
    day_mad = g.groupby("date")["gap_pct"].transform(
        lambda x: np.median(np.abs(x - np.median(x))))
    g["day_median_gap_pct"] = day_med
    g["scaled_dev"] = (g["gap_pct"] - day_med) / (1.4826 * day_mad)
    g["flag_bool"] = g["scaled_dev"] > K_FLAG
    g["low_cov_bool"] = g["coverage_share"] < LOW_COVERAGE_SHARE

    g = g.sort_values(["SOURCE_KEY", "date"]).reset_index(drop=True)

    def streak(flags):
        out, run = [], 0
        for f in flags:
            run = run + 1 if f else 0
            out.append(run)
        return out

    g["consecutive_days_flagged"] = g.groupby("SOURCE_KEY")["flag_bool"].transform(
        lambda s: pd.Series(streak(list(s)), index=s.index))

    g = g.sort_values(["date", "gap_pct", "SOURCE_KEY"], ascending=[True, False, True])
    g["rank"] = g.groupby("date").cumcount() + 1

    g["flagged"] = np.where(g["flag_bool"], "yes", "no")
    g["low_coverage"] = np.where(g["low_cov_bool"], "yes", "no")
    g["tie_break_rule"] = TIE_RULE

    cols = ["date", "SOURCE_KEY", "rank", "gap_pct", "coverage", "day_timestamps",
            "coverage_share", "day_median_gap_pct", "scaled_dev",
            "flagged", "low_coverage", "consecutive_days_flagged", "tie_break_rule"]
    rank_out = g[cols].copy()
    rank_out.to_csv(os.path.join(OUT, "daily_ranking.csv"), index=False)
    log(f"P4 inverter-days: {len(rank_out)}  flagged: {int(g['flag_bool'].sum())}  "
        f"low coverage: {int(g['low_cov_bool'].sum())}  ({time.perf_counter() - t:.4f} s)")

    # ---------------- P5: cause features and labels ----------------
    t = time.perf_counter()
    u2 = exp_rows[exp_rows["expected_valid"]].copy()
    u2["hour"] = u2["timestamp"].dt.hour
    u2["shortfall"] = u2["expected"] - u2["AC_POWER"]
    u2["below"] = u2["AC_POWER"] < u2["expected"]
    u2["dropout"] = (u2["AC_POWER"] == 0) & (u2["expected"] > 0)

    h1 = u2.groupby(["SOURCE_KEY", "date"])["below"].mean().rename("H1_share_below")
    h3 = u2.groupby(["SOURCE_KEY", "date"])["dropout"].sum().rename("H3_dropout_count")
    cov = u2.groupby(["SOURCE_KEY", "date"]).size()
    h5 = (1 - cov / cov.index.get_level_values("date").map(day_ts)).rename("H5_missing_share")

    hour_sum = u2.groupby(["SOURCE_KEY", "date", "hour"])["shortfall"].sum()
    pos = hour_sum.where(hour_sum > 0)
    h2 = (pos.groupby(level=[0, 1]).max() / pos.groupby(level=[0, 1]).sum()
          ).rename("H2_worst_hour_share")

    piv_dc = table.pivot_table(index="timestamp", columns="SOURCE_KEY", values="DC_POWER")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = piv_ac.to_numpy() / piv_dc.to_numpy()
    ratio = np.where(piv_dc.to_numpy() > 0, ratio, np.nan)

    peer_med = np.full(ratio.shape, np.nan)
    for i in range(ratio.shape[0]):
        row = ratio[i]
        if np.sum(~np.isnan(row)) - 1 < MIN_PEERS:
            continue
        for j in range(ratio.shape[1]):
            if np.isnan(row[j]):
                continue
            others = np.delete(row, j)
            others = others[~np.isnan(others)]
            peer_med[i, j] = np.median(others)

    ti2 = u2["timestamp"].map(ts_pos).to_numpy()
    sj2 = u2["SOURCE_KEY"].map(col_pos).to_numpy()
    u2["own_ratio"] = ratio[ti2, sj2]
    u2["peer_ratio"] = peer_med[ti2, sj2]
    u2["rel_dev_pct"] = ((u2["own_ratio"] / u2["peer_ratio"]) - 1).abs() * 100
    h4 = u2.groupby(["SOURCE_KEY", "date"])["rel_dev_pct"].mean().rename("H4_ratio_dev_pct")

    feat = pd.concat([h1, h2, h3, h4, h5], axis=1).reset_index()

    fl = rank_out[rank_out["flagged"] == "yes"][
        ["date", "SOURCE_KEY", "rank", "gap_pct", "coverage_share", "scaled_dev"]].copy()
    work = fl.merge(feat, on=["SOURCE_KEY", "date"], how="left")
    work["m_H1"] = work["H1_share_below"] > THR_H1
    work["m_H2"] = work["H2_worst_hour_share"] > THR_H2
    work["m_H3"] = work["H3_dropout_count"] >= THR_H3
    work["m_H4"] = work["H4_ratio_dev_pct"] > THR_H4
    work["m_H5"] = work["H5_missing_share"] > THR_H5

    def labels_of(row):
        hit = [k for k in ["H1", "H2", "H3", "H4", "H5"] if row["m_" + k]]
        return ";".join(hit) if hit else "UNDETERMINED"

    work["labels"] = work.apply(labels_of, axis=1)
    work["label_count"] = work["labels"].apply(
        lambda s: 0 if s == "UNDETERMINED" else len(s.split(";")))

    cause_out = work[["date", "SOURCE_KEY", "rank", "gap_pct", "coverage_share", "scaled_dev",
                      "H1_share_below", "H2_worst_hour_share", "H3_dropout_count",
                      "H4_ratio_dev_pct", "H5_missing_share",
                      "labels", "label_count"]].copy()
    for c in ["H2_worst_hour_share", "H4_ratio_dev_pct", "H1_share_below",
              "H5_missing_share", "gap_pct", "scaled_dev"]:
        cause_out[c] = cause_out[c].round(6)
    cause_out.to_csv(os.path.join(OUT, "cause_hypotheses.csv"), index=False)
    log(f"P5 flagged inverter-days: {len(cause_out)}  "
        f"labels: {dict(cause_out['labels'].value_counts())}  ({time.perf_counter() - t:.4f} s)")

    # ---------------- P7a: final table ----------------
    t = time.perf_counter()
    final = rank_out[["date", "SOURCE_KEY", "rank", "gap_pct", "coverage",
                      "day_timestamps", "coverage_share", "day_median_gap_pct",
                      "scaled_dev", "flagged", "low_coverage",
                      "consecutive_days_flagged"]].copy()
    hyp = cause_out[["date", "SOURCE_KEY", "H1_share_below", "H2_worst_hour_share",
                     "H3_dropout_count", "H4_ratio_dev_pct", "H5_missing_share",
                     "labels", "label_count"]]
    final = final.merge(hyp, on=["date", "SOURCE_KEY"], how="left")
    final["labels"] = final["labels"].fillna("not_flagged")
    final["label_count"] = final["label_count"].fillna(-1).astype(int)
    final["tie_break_rule"] = TIE_RULE
    final.to_csv(os.path.join(OUT, "final_daily_ranking.csv"), index=False)
    log(f"P7 final rows: {len(final)}  ({time.perf_counter() - t:.4f} s)")

    # ---------------- P7b: Plot A ----------------
    t = time.perf_counter()
    one = u2[(u2["SOURCE_KEY"] == PLOT_A_INVERTER) &
             (u2["date"] == pd.Timestamp(PLOT_A_DATE).date())].sort_values("timestamp")
    gap_val = float(final[(final["SOURCE_KEY"] == PLOT_A_INVERTER) &
                          (final["date"] == pd.Timestamp(PLOT_A_DATE).date())]["gap_pct"].iloc[0])
    lab = str(final[(final["SOURCE_KEY"] == PLOT_A_INVERTER) &
                    (final["date"] == pd.Timestamp(PLOT_A_DATE).date())]["labels"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(one["timestamp"], one["expected"], label="expected (median of the other 21 inverters)",
            linewidth=2, color="#1f77b4")
    ax.plot(one["timestamp"], one["AC_POWER"], label="actual", linewidth=2, color="#d62728")
    ax.fill_between(one["timestamp"], one["AC_POWER"], one["expected"],
                    where=one["expected"] >= one["AC_POWER"],
                    color="#d62728", alpha=0.15, label="shortfall")
    ax.set_title(f"{PLOT_A_INVERTER} vs its peers, {PLOT_A_DATE}\n"
                 f"gap_pct = {gap_val:.4f}   labels: {lab}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("AC_POWER, values as given in the file")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_a_actual_vs_expected.png"), dpi=150)
    plt.close(fig)
    log(f"P7 plot A: {PLOT_A_INVERTER} {PLOT_A_DATE}, {len(one)} timestamps  "
        f"({time.perf_counter() - t:.4f} s)")

    # ---------------- P7c: Plot B ----------------
    t = time.perf_counter()
    grid = final.pivot_table(index="SOURCE_KEY", columns="date", values="gap_pct")
    grid = grid.sort_index()
    days = sorted(grid.columns)
    vmax = float(np.percentile(np.abs(grid.to_numpy()[~np.isnan(grid.to_numpy())]), 99))

    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d.strftime("%d-%m") for d in days], rotation=90, fontsize=7)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index, fontsize=8)
    ax.set_title(f"gap_pct for every inverter and every day\n"
                 f"red = below peers, blue = above peers, colour clipped at +/-{vmax:.2f}")
    ax.set_xlabel("day (2020)")
    ax.set_ylabel("inverter (SOURCE_KEY)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("gap_pct")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_b_gap_grid.png"), dpi=150)
    plt.close(fig)
    log(f"P7 plot B: {grid.shape[0]} inverters x {grid.shape[1]} days  "
        f"({time.perf_counter() - t:.4f} s)")

    # ---------------- manifest ----------------
    log("")
    log("MANIFEST")
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name)
        if os.path.isfile(p):
            log(f"  {name:<38} {os.path.getsize(p):>9} bytes")
    log("")
    log(f"total wall time: {time.perf_counter() - t_start:.2f} s")


if __name__ == "__main__":
    main()
