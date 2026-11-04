# SolarGuard AI

**Detect and rank underperforming solar inverters using multi-model peer comparison — no fault labels or weather data required.**

> 🤖 **This is a vibecoded project.** The entire codebase — data pipeline, three ML models, Streamlit dashboard, and all analysis — was built through AI-assisted pair programming using conversational prompts. Every threshold and design decision was proposed by the AI, reviewed by the team, and explicitly approved before being applied.

Built by **Team TECHNOCRATS** (Team ID: U091EK2H) for the **AI Sansad Technical MUN on AI & Innovation**, Round 2 AI/ML Model Challenge, Bharati Vidyapeeth's College of Engineering, New Delhi.

---

## The Problem

A solar plant has 22 inverters generating power readings every 15 minutes over 34 days. The task: **which inverters produced less than their neighbours, and why might that be?**

There are no fault labels, no weather data (irradiance, temperature), and no data dictionary in the dataset. SolarGuard AI solves this with unsupervised peer comparison and multi-model consensus.

## The Approach — Three Independent Models

### Model 1: Real-Time Spatial Peer Median (baseline)
For each inverter at each timestamp, the **median AC_POWER of the other 21 inverters** serves as the expected value. A daily `gap_pct` measures how far below peers the inverter fell. Outliers are flagged using MAD-based scaled deviation (k = 4).

### Model 2: k-NN Curve Memory
A memory bank of **696 clean daily generation curves** (51 daytime slots each). Each inverter-day is scored by its Euclidean distance to the 3 nearest normal historical profiles.

### Model 3: Random Forest Regressor
A 100-tree Random Forest trained on diurnal time cycles and peer distributions to predict AC_POWER. Test R² = **96.60%** on unseen days (Days 25–34). Daily residuals are flagged with the same MAD outlier rule.

### Multi-Model Consensus
Across all **748 inverter-days**, the three models achieve **91.04% consensus agreement**, isolating **25 triple-confirmed flagged days** where all three algorithms independently agree on underperformance.

---

## Quick Start

### Prerequisites

- Python 3.10+
- pandas, numpy, matplotlib
- scikit-learn (for the Random Forest model)
- streamlit (for the interactive dashboard)

### Rebuild All Outputs from Scratch

Place `Plant_1_Generation_Data.csv.zip` in the project root, then:

```bash
python outputs/run_all.py
```

This rebuilds all tables and plots from the original zip in ~2.25 seconds.

### Launch the Dashboard

```bash
python -m streamlit run outputs/dashboard.py
```

Opens at `http://localhost:8501` with four interactive tabs covering all three models and the consensus view. Supports drag-and-drop upload of new `.csv` or `.zip` datasets.

---

## Key Results

| Metric | Value |
|---|---|
| Total inverter-days analysed | 748 (22 inverters × 34 days) |
| Inverter-days flagged (baseline) | 52 (6.95%) |
| Triple-confirmed flagged days | 25 |
| Triple-confirmed normal days | 656 |
| Three-model consensus | 91.04% |
| Full pipeline rebuild time | 2.25 s |
| Peak memory | 19.50 MiB |

Two inverters account for 45 of the 52 baseline flags. 14 of the 22 inverters are never flagged.

---

## Cause Hypotheses

Each flagged inverter-day is assigned one or more hypothesis labels:

| Label | Pattern | Possible Cause |
|---|---|---|
| H1 | Below expected on >80% of daytime timestamps | Gradual loss (e.g. dirt, degradation) |
| H2 | Worst hour accounts for >85% of shortfall | Shading or time-dependent limit |
| H3 | ≥2 timestamps at zero while peers produce | Outage or shutdown |
| H4 | AC/DC ratio deviates >0.10% from peers | Conversion or measurement difference |
| H5 | >5% of daytime timestamps missing | Logging or communication gap |

> **Every cause is a hypothesis.** None can be confirmed from this dataset alone.

---

## Project Structure

```
SolarInverterPS/
├── README.md                              # This file
├── AGENTS.md                              # AI agent behaviour rules
├── PROJECT_STATE.md                       # All approved decisions and results
├── PROJECT_HANDOFF.md                     # Comprehensive handoff document
├── SolarGuard_Brief_and_Architecture.md   # Event brief and system architecture
│
├── outputs/                               # Pipeline outputs & dashboard
│   ├── clean_table.csv                    # 36,870 daytime rows
│   ├── expected_table.csv                 # Peer expected values
│   ├── daily_ranking.csv                  # Daily gap % and MAD flags
│   ├── cause_hypotheses.csv               # 52 flagged days with H-labels
│   ├── final_daily_ranking.csv            # Complete joined daily table
│   ├── plot_a_actual_vs_expected.png      # Sample shortfall timeseries
│   ├── plot_b_gap_grid.png                # 22×34 heatmap
│   ├── run_all.py                         # End-to-end rebuild script
│   ├── summary.md                         # One-page executive summary
│   ├── demo_script.md                     # 2-minute spoken walkthrough
│   └── dashboard.py                       # Streamlit multi-model app
│
├── memoryNN/                              # k-NN Curve Memory Model
│   ├── knn_memory.py                      # Memory bank construction
│   ├── run_knn_steps3_4.py                # Anomaly scoring
│   ├── knn_memory_ranking.csv             # 748 rows of memory distances
│   └── plot_knn_memory_example.png        # 3-nearest curve retrieval
│
├── random_forest/                         # Random Forest Regressor
│   ├── rf_model.py                        # 100-tree RF regressor
│   ├── rf_model_ranking.csv               # 748 rows with RF flags
│   └── plot_rf_prediction_example.png     # Predicted vs actual plot
│
└── base/                                  # Preserved baseline snapshot
    ├── PROJECT_STATE.md
    ├── SolarGuard_Brief_and_Architecture.md
    └── outputs/                           # Mirrored baseline outputs
```

---

## Dataset

**Source:** `Plant_1_Generation_Data.csv.zip` (872 KB compressed, ~4.8 MB extracted).

| Property | Value |
|---|---|
| Rows | 68,778 |
| Columns | `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD` |
| Plant | Single plant (ID 4135001) |
| Inverters | 22 distinct `SOURCE_KEY` values |
| Date range | 2020-05-15 to 2020-06-17 (34 days) |
| Interval | 15-minute readings |
| Missing cells | 0 |

**Not in the file:** fault labels, irradiance, temperature, voltage, current, rated capacity, or a data dictionary.

---

## Approved Rules & Thresholds

All thresholds were derived from data distributions and approved by the team:

| Rule | Approved Value |
|---|---|
| Daytime definition | Timestamp where >50% of active inverters have AC_POWER > 0 |
| Gap filling | None (no synthesised rows, no changed values) |
| Minimum peers | n_peers ≥ 10 |
| Flag threshold (k) | scaled_dev > 4.0 |
| Low coverage | coverage_share < 95% |
| k-NN neighbours | k = 3, distance cutoff = 398.459 (95th percentile) |
| Random Forest | 100 trees, max_depth = 12, time-split at Day 24/25 |

---

## Limitations

- **No fault labels exist.** "Flagged" means ranked low vs. peers by an approved rule — not proof of a fault.
- **Units are unknown.** Values are reported as given in the file.
- **No weather data.** Dirt, shading, and degradation cannot be separated.
- **Plant-wide problems are invisible.** If every inverter drops together, gap_pct stays near zero.
- **34 days, one plant.** Nothing here generalises to another site or season.
- **Every cause is a hypothesis**, not a finding.

---

## Dependencies

| Package | Version Used |
|---|---|
| Python | 3.13.14 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| streamlit | 1.66.1 |

---

## Team

**TECHNOCRATS** — Team ID U091EK2H  
Leader: Jitinpal Singh  
Members: 3

---

## License

This project is licensed under the [MIT License](LICENSE). The dataset is provided by the AI Sansad competition organisers.

