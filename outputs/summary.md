# SolarGuard AI — summary

One plant, 34 days, 22 inverters. Every number below is copied from a saved file in `outputs/` or from a run shown in this session.

## What we did

1. Read the original zip without changing it. 68,778 rows, 7 columns, 2020-05-15 to 2020-06-17. Re-checked 36 claims about the file in Phase 1 and found 0 differences.
2. Kept daytime only: 1,680 of the 3,158 timestamps present. Result: `clean_table.csv`, 36,870 rows.
3. For each inverter at each timestamp, computed what its peers produced: the median of the other inverters at that same moment. Result: `expected_table.csv`, an expected value on 36,846 of 36,870 rows.
4. Measured each inverter's day against its peers: `gap_pct`, the share of expected output that is missing. Result: 748 inverter-days.
5. Ranked the inverters each day, flagged the unusual ones, and attached cause hypotheses to flagged days only.

## Approved rules, with exact values

| rule | value |
|---|---|
| Daytime | a timestamp is daytime when more than half of the inverters present are above zero |
| Gap filling | none — no synthesized rows, no changed values |
| Minimum peers | `n_peers >= 10`; below that the expected value stays empty |
| `expected(i,t)` | median of `AC_POWER` over all other inverters with a reading at `t` |
| `gap_pct(i,d)` | `(sum expected - sum actual) / sum expected * 100`; positive = below peers |
| Flag | `(gap_pct - day median) / (1.4826 x day MAD) > 4` |
| Low coverage | `coverage / daytime timestamps that day * 100 < 95` |
| Rank | 1 = largest `gap_pct`; ties by `SOURCE_KEY` ascending (no tie occurred) |
| H1 | share of the day's timestamps below expected `> 0.80` |
| H2 | worst positive hour's share of the day's positive shortfall `> 0.85` |
| H3 | count of timestamps at zero while peers produce `>= 2` |
| H4 | mean relative AC/DC gap versus peers `> 0.10 %` |
| H5 | share of the day's daytime timestamps missing `> 0.05` |

## What the results show

- `gap_pct` over 748 inverter-days: median -0.2501, 95th percentile 7.7320, largest 45.7892.
- **52 inverter-days flagged** (6.95%), spread over 25 of the 34 days. Maximum run for one inverter: 8 days.
- **4 inverter-days marked low coverage**, all on 2020-05-19 at 38 of 44 timestamps.
- Cause hypotheses on the 52 flagged days: **H1 on 45**, **H3 on 9** (4 carry both), **UNDETERMINED on 2**. No flagged day met H2, H4 or H5.
- Persistence: `bvBOhCH3iADSZry` flagged 25 days with an 8-day streak; `1BY6WEcLGh8j5v7` 20 days with a 3-day streak. 14 of the 22 inverters are never flagged.

## Checks, and what they do not prove

| check | result | does NOT prove |
|---|---|---|
| Mean of peers instead of median | per-day rank correlation min 0.979673, median 0.997741; 2 of 52 flags change | that either ranking is correct |
| Repeat days | 2 inverters account for 45 of the 52 flags | why those days repeat |
| Rule derived on days 1–24, applied to days 25–34 | derived threshold 4.977069 gives 11 flags in the later window against 15, so 4 change | that the future resembles the past |
| Rebuild from the original zip | `run_all.py` reproduces all four earlier tables byte-for-byte (identical SHA256) | that the input data is complete |
| Run time and memory | 2.25 s for a full rebuild; peak Python memory 19.50 MiB | speed on another machine |

There are no truth labels in this file, so **no accuracy, precision or recall figure is reported or implied anywhere.**

## Limitations

- **No fault labels exist.** "Flagged" means ranked low against peers by an approved rule. It is not proof of a fault.
- **Units are unknown.** All values are stated "as given in the file". Percentages are ratios of two file values, so units cancel.
- **No weather data.** No irradiance, temperature, voltage, current or rated capacity is in the file, so a passing cloud and a dirty panel cannot be separated.
- **One plant, 34 days.** Nothing here generalises to another site or season.
- **Plant-wide problems are invisible.** If every inverter drops together, the median drops with them and `gap_pct` stays near zero.
- **Every cause is a hypothesis.** H1 suggests gradual loss such as dirt or degradation; H3 suggests an outage or shutdown. Neither can be confirmed from this file.
- **3,030 missing 15-minute slots were left missing**, never filled.

## What we do NOT know

- Whether any inverter is actually faulty.
- What caused the low output on the flagged days.
- Whether the 106 timestamps absent for every inverter are a logging gap or a real stop.
- What the AC/DC ratio near 0.098 physically represents.
- Whether the two `UNDETERMINED` days share a cause with the labelled ones.

## Files

| file | contents |
|---|---|
| `clean_table.csv` | daytime readings, original values untouched, `was_filled` / `was_missing` flags |
| `expected_table.csv` | actual, expected, `n_peers`, `expected_valid` |
| `daily_ranking.csv` | one row per inverter-day: rank, gap_pct, coverage, flagged, low_coverage, streak |
| `cause_hypotheses.csv` | the 52 flagged days with all five measurements and their labels |
| `final_daily_ranking.csv` | ranking and hypotheses together, one row per inverter-day |
| `plot_a_actual_vs_expected.png` | one flagged day, actual against expected |
| `plot_b_gap_grid.png` | every inverter against every day |
| `run_all.py` | rebuilds every table and both plots from the original zip |
| `demo_script.md` | the 2-minute spoken walkthrough |
