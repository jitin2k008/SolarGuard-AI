# SolarGuard AI — 2-minute demo script

Spoken words only. Every number below is copied from `outputs/final_daily_ranking.csv` or `outputs/cause_hypotheses.csv`.

---

**[0:00 — 0:15] The problem**

> "One solar plant. 22 inverters. 34 days. Every 15 minutes we have a reading for each one.
> The question we answer is simple: **which inverters produced less than their neighbours, and why might that be?**"

---

**[0:15 — 0:50] The method, in one idea**

> "We do not have a fault list, and we do not have weather data. So we compare each inverter to the other 21 at the exact same moment.
>
> For every timestamp we take the **median** of the other inverters. That is our expected value. Then for each inverter and each day we compute `gap_pct` — the share of expected output that is missing. Positive means it produced less than its peers.
>
> The result is **748 inverter-days** — 22 inverters times 34 days."

---

**[0:50 — 1:15] The ranking and the flag**

> "Every day we rank all 22 from the biggest gap down. Then we flag the unusual ones.
>
> A day is **flagged** when its gap is more than 4 scaled deviations above that day's typical gap. That is `k = 4`, a value our user approved after seeing the distribution.
>
> It flags **52 of the 748 inverter-days** — 6.95 percent — spread across 25 of the 34 days."

---

**[1:15 — 1:45] The picture on screen** *(show `plot_a_actual_vs_expected.png`)*

> "This is `bvBOhCH3iADSZry` on 14 June 2020. Red is what it actually produced, blue is what the other 21 produced at the same moments.
>
> Its gap that day was **45.789197** — rank 1 of 22. The blue line carries on through the middle of the day while the red line sits at zero for 12 timestamps. That is why we labelled it **H1 and H3** — below expected on 96.08 percent of its readings, with 12 dropouts."

---

**[1:45 — 2:00] The honest ending**

> "Two things we will not claim. This is **not** a fault list — there are no fault labels in this file, so we cannot report accuracy. And every cause is a **hypothesis**, not a finding.
>
> What we can show is a ranking that rebuilds itself from the original file in **2.25 seconds**, and that two inverters hold 45 of the 52 flags while 14 are never flagged at all."

---

## If a judge asks a question

| question | answer |
|---|---|
| "Is this accurate?" | "There are no truth labels in this file, so accuracy cannot be stated. What we show is repeatability: the rebuild reproduces every earlier table byte-for-byte." |
| "Why median and not mean?" | "We tested both. Per-day rank correlation was 0.979673 at the lowest and 0.997741 in the middle, and only 2 of the 52 flags changed." |
| "Why 4?" | "It is an approved value, not a fact. At the 95th percentile of the earlier 24 days the threshold would be 4.977069, which gives 11 flags in the last 10 days instead of 15." |
| "So which inverter is faulty?" | "We do not say faulty. We say flagged — ranked low against peers by an approved rule. `bvBOhCH3iADSZry` was flagged on 25 days, which is a pattern worth a physical inspection." |
| "Why are units not shown?" | "The file does not state them, so we report values as given in the file. The percentages are ratios of two file values, so units cancel." |
| "What can this miss?" | "If every inverter drops together, the median drops with them and the gap stays near zero. Plant-wide problems are invisible to this method." |

---

## Do not say

- "faulty", "broken", "failed" — say **flagged**
- "kW", "kWh", "MW" — say **as given in the file**
- "accuracy", "precision", "recall" — there are no labels
- "it failed because of dirt" — say **[HYPOTHESIS]** and name the data that would confirm it
