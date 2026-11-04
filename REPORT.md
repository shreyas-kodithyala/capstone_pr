# Project report: Federated, onboard health monitoring for multi-satellite systems

PES University Capstone 122. Team of 4, guide Dr. Richa Sharma.
Report date: 2 November 2026 (first review: 3 November 2026).

This report records everything done so far, explains how the detector works, gives every result
with its caveats, and sets out the planned next phase. Results that were not run are marked as
not run. Numbers come from the result files listed in section 9.

---

## Contents

1. Problem statement
2. Background: why this problem matters (industry research)
3. Data: the ESA Anomaly Detection Benchmark (ESA-ADB)
4. The detector, end to end
5. How results are scored and kept honest
6. Results so far
7. Bugs found and fixed
8. The next phase: onboard + federated + simulated fleet
9. Where everything is (files and folders)
10. Open items and risks

---

## 1. Problem statement

> Satellite health monitoring today relies on static onboard limit checks and on ground-based ML
> over downlinked telemetry. That approach is slow, limited by bandwidth, prone to false alarms,
> and unable to learn from faults seen elsewhere in a fleet. This project develops an onboard,
> federated health-monitoring architecture for multi-satellite systems.
>
> Each satellite runs a lightweight (int8, ~0.2 MB) graph-temporal anomaly detector. It models
> dependencies between sensors and telecommands, flags the affected component, and triggers a
> tiered FDIR (fault detection, isolation and recovery) response. Satellites improve their models
> collaboratively by exchanging model updates, never raw telemetry, over inter-satellite links.
> Updates are aggregated by a rotating leader satellite with robust filtering of faulty nodes.
> Ground operators review alarms, and their verdicts feed back into training.
>
> Detection performance is validated on real spacecraft telemetry (ESA Anomaly Detection
> Benchmark), where the detector exceeds the published baselines. Fleet learning, degradation
> early warning and closed-loop FDIR are evaluated in a Basilisk-based multi-satellite simulation.
> Onboard feasibility is measured by running the quantized detector on an ARM Cortex-A53.
> Integration with NASA cFS flight software and LEON-class processors is future work.

**How the two halves fit together:**

- **Real data (ESA-ADB) is the validation.** On telemetry from two real ESA spacecraft, the
  detector beats the published benchmarks, and the operator loop improves it month by month.
- **The simulator (Basilisk) is the system demonstration.** It tests what real single-satellite
  data cannot: a fleet of satellites of the same design, federated learning between them,
  controlled degradation, and closed-loop recovery, all with exact ground truth.

A scope check was done (section 8.5). The original wording claimed the full system runs as cFS
flight software on LEON3. That is weeks of C porting, so it was moved to future work.

---

## 2. Background: why this problem matters

These points come from a literature and industry search (2 November 2026). It was a search-level
review, not a systematic one. Any "first to do X" claim must be checked properly (Google Scholar,
IEEE Xplore) before it goes into a paper.

| # | Problem | Evidence |
|---|---|---|
| P1 | Reaction wheels are among the most common failures, and about a quarter of in-orbit anomalies end in total loss | 2024 statistical review of on-orbit anomalies (ScienceDirect S0094576524000560). The leading wheel failure cause is bearing lubrication / friction growth (PMC9919697) |
| P2 | Fixed-threshold alarms miss slow degradation and subtle in-range faults. Ground-in-the-loop isolation delays recovery | AI onboard FDIR study (SmallSat, USU digital commons 5708). ESA states traditional FDIR is weak at isolation and at combined faults (ESA Nebula: "Machine learning-based on-board autonomy, failure prognostics and detection") |
| P3 | Floods of false alarms overwhelm operators and block AI adoption. In mega-constellations, humans cannot review every data point | SpaceOps 2025 #152, SpaceOps 2023 #336, arXiv 2601.12667 |
| P4 | Identical satellites diverge over time, so one shared baseline does not fit all | Iridium NEXT operations paper (AIAA 2018-2708) |
| P5 | Labels are scarce and incomplete (many anomaly records lack cause or date), and new satellites have no history | Only 55% of anomalies in the 2024 review even report a date |
| P6 | Ageing and drift cause false alarms, and retraining with expert feedback is unstudied | arXiv 2311.10421: "no study on the effects of blind and informed retraining" |
| P7 | Operators need to trace why an alarm fired | arXiv 2604.08424 |

**What already exists, so we do not claim it:**

- Federated learning for satellite telemetry has been published, but the results are weak. One
  attention-LSTM paper reports AUC 0.51, close to random. Others use synthetic data, or split one
  satellite's history into time slices and call them clients.
- Reaction-wheel prognostics (remaining useful life) is well studied: Kalman and particle filters,
  LSTMs. Wheel friction detection and categorization reaches about 95% (arXiv 2509.04060).
- Operator-feedback retraining has been shown for connected vehicles and manufacturing, but not
  for spacecraft fleets (arXiv 2607.08373).

**Where this project is new:** the combination, validated on real data:

- a detector that beats the published ESA-ADB benchmarks;
- personalised federated learning between satellites of the same design;
- operator feedback shared across the fleet ("one operator's verdict improves every satellite");
- alarms that separate commanded changes from real faults;
- onboard int8 feasibility.

---

## 3. Data: ESA Anomaly Detection Benchmark (ESA-ADB)

ESA-ADB is real housekeeping telemetry from ESA missions, published with expert labels and an
official scoring harness (Kotowski et al., arXiv 2406.17826; Zenodo 15237121;
github.com/kplabs-pl/ESA-ADB). Mission 3 is excluded by the benchmark authors, and we follow that.

| | Mission 1 (M1) | Mission 2 (M2) |
|---|---|---|
| Channels | 76 (58 scored, "Target=YES") | 100 (47 scored) |
| Grid | 30 s | 18 s |
| Length | 14 years (train 2000–2007, test 2007–2014) | 3.5 years |
| Telecommands | 698 (priority ≥2: 30; priority 3: 11) | 123 (priority 3: 4) |
| Events | 200 (118 anomalies, 78 rare events, 4 gaps) | 644 (31 anomalies, 613 rare events) |
| Lightweight subset (official) | channels 41–46 (= subsystem 5) | channels 18–28 (part of subsystem 1) |
| Test events, lightweight | 65 (29 anomalies, 36 rare events) | 350 |

**Key data finding (from the earlier phase).** ESA's preprocessing fills missing samples with the
previous value ("zero-order hold"). On M1 this hides about 30% of the anomalies: rows where no new
sample arrived (`count == 0`) are anomalous 56.7% of the time, 31× the base rate.

- A forecaster sees a flat, perfectly predictable line and detects nothing.
- We recovered label-free availability features from the raw files: `count` (samples in the
  bucket), a gap flag, and `tsince` (time since the last real sample).
- We added a silence detector on top of them.
- This addresses ESA's requirement R8. It is also why the published forecasting baselines
  underperform on M1.

**Subsystems.** Subsystem names are consistent across the two missions (the ESA paper, Table 1).
Shared subsystems are S1, S5 and S6.

| Pair | M1 | M2 | Behaviour |
|---|---|---|---|
| S5 (cleanest pair) | 6 channels | 19 channels | same name and unit, but M1 is noisy and M2 holds values |
| S6 (most alike) | 42 channels | 12 channels | slow, smooth and highly redundant in both missions |

---

## 4. The detector, end to end

The detector has two layers: a neural forecaster that predicts each channel's next value, and a
detection stack that turns forecast errors into alarms.

### 4.1 Inputs (per 30 s row)

- **Each channel node** gets 3 numbers:
  - the value, z-scored with training-split mean and std;
  - a gap flag (`count == 0`);
  - `log10(1 + tsince) / 7`.
- **Each telecommand** gets 5 numbers, computed from its firing times up to that row:
  - recency, `exp(-dt/1800)`;
  - log time since the last firing;
  - log number of firings in the last 1 h, 6 h and 24 h.
- Telecommands with priority ≥2 are used: M1 30, M2 123. The subsystem models use ≥3.
- Labels are never inputs. They are only used to leave labelled rows out of the training loss.

### 4.2 The forecaster: GDN2 (`core/model.py`)

This is a Graph Deviation Network (Deng and Hooi, 2021), upgraded. It has 202,018 parameters.

| Part | What it does | Shared in federated learning? |
|---|---|---|
| Channel embedding (64-d per channel) | Identity of each sensor; drives the graph | **No:** local to each spacecraft |
| TCN encoder | Causal dilated temporal convolutions, 8 layers, 64 filters, kernel 5, GELU, dropout 0.1, residual connections. Receptive field 1,021 rows (8.5 h on M1). Each channel is encoded separately | Yes |
| Learned directed graph (STGLR-style "dir_anti") | Two projections of the embeddings form an antisymmetric score, so edges point one way (A drives B). It is **dynamic**: recomputed at every time step from the current encoder state. Each channel keeps its top-k neighbours (k = min(N−1, 10)) | Yes (the projections) |
| GATv2 attention | Each channel attends over its neighbours' encoder states; the graph adds an edge-strength bias | Yes |
| Telecommand path | Each channel queries the commands with cross-attention, with a "nothing happened" slot. A learned gate (starting near 0.05) decides how much command context enters | Encoder, query and gate yes; command embedding no |
| Output head | Per-channel MLP on [embedding × context, context] → next value | Yes |

- **Federated split:** `shared_state()` returns every tensor except `embedding.*` and
  `cmd_emb.*`. All shared tensors have shapes that do not depend on the channel count, so models
  for different spacecraft can be averaged. This was checked: 41 shared tensors load cleanly from
  M2 into M1.
- **Training:**
  - masked MSE (labelled rows weighted 0);
  - Adam, learning rate 1e-3;
  - 300 batches per epoch, up to 30 epochs;
  - early stopping (patience 8) on the last 15% of the training period;
  - segments of 2 × receptive field.

### 4.3 Detection stack (`core/period.py`, `core/detect.py`)

Every statistic uses the past only. A decision may look at most 70 rows ahead, which is ESA
Telemanom's batch size.

| Detector | What it does |
|---|---|
| **B: robust z of forecast error** | Errors are smoothed (EWMA, spans 1/3/15/60/240 or 15). Each is z-scored against a **past-only** baseline: the median and IQR of the previous window W (1 day … all past), refreshed every 6 h from histograms. Alarm if z > threshold |
| **A: ESA Telemanom (sliding)** | ESA's own Telemanom error rules, reproduced faithfully: windows of 30 × 70 rows, decide only the newest 70, inverse errors, **pruning** (M1 only) |
| **C: silence** | No telemetry for ≥ L rows (the availability finding) |
| **STD-k** | Raw value outside training mean ± k·std |
| **Drift** | 1-day vs 30-day (or 1-week vs 90-day) shift |

**Post-processing:**
- hysteresis and cross-channel extension (other channels above a fraction of the threshold join an
  alarm);
- onset refinement;
- merging nearby alarms (`close`) and dropping short ones (`minlen`).

**Alarm typing:** an alarm that starts within a window after a telecommand is labelled a
"commanded change" instead of an anomaly.

**Per-mission detection policy (a finding in itself).** The same encoder needs opposite rules:

- Telemanom pruning gives M1 0.853 but M2 0.014.
- An absolute z threshold gives M2 0.964 but M1 0.137.
- So each spacecraft keeps its own detector setting. This is "personalised detection policy", the
  basis of the federated design.

| Chosen setting | Mission 1 lightweight | Mission 2 lightweight |
|---|---|---|
| B threshold | absolute z 20, window 6 months, multi-span | absolute z 20, window "all", span 15 |
| A Telemanom | on: p 0.05, span 105, min error 0.007, back-reach 25 | off |
| C silence | L = 10 | off |
| STD-k | k = 8 | off |
| Post-processing | close 60, cross 0.3, minlen 10 | close 60, cross 0.3, minlen 10 |
| Typing window | 3600 s, priority 3 | 600 s, priority 2 |

### 4.4 Choosing detector settings without touching the test split (`core/select.py`)

- Two **validation folds** inside the training years:
  - M1: fold A 2003–2004, fold B 2005–2006;
  - M2: two 6-month folds.
- A model trained only on data before each fold forecasts that fold (unseen data).
- A coordinate search picks the setting that maximises pooled F0.5 over A + B. Among settings
  within 0.02 of the best, it prefers the one whose worse fold is best.
- The test split is scored **once**, at the end.
- Fold scores predicted the test well: M1 folds 0.894 → test 0.866.

### 4.5 The pipeline as code

`core/train.py` (train; write errors) → `core/select.py` (choose a setting on folds) →
`core/evaluate.py` / `evaluate_par.py` (one test scoring, all ESA aspects, events caught,
comparison with fixed limits). All under `ESA/Main/new_runs/`.

---

## 5. How results are scored and kept honest

- **Official metric:** ESA's event-wise F0.5 with precision corrected for false-alarm time:
  `Pre = TPe/(TPe+FPe) × (1 − FPt/Nt)`, β = 0.5 (precision weighted 4×). Events are "Anomaly" and
  "Rare Event".
- `core/esa.py` uses ESA's own metric classes (vendored, unchanged). Our harness replica matched
  the official Docker scores to 1e-12 in the earlier phase.
- **All five ESA aspects are reported, not just the headline:**
  - event-wise F0.5;
  - channel-aware F0.5 and subsystem-aware F0.5;
  - alarm precision;
  - ADTQC (alarm timing quality);
  - affiliation F0.5.
- **Rules followed throughout:**
  - never feed `is_anomaly` as an input;
  - no thresholds tuned on the test split, and no oracle scores;
  - all detection is past-only;
  - one test scoring per configuration;
  - Telemanom pruning stays on M1 and off M2;
  - no single global score broadcast to all channels;
  - Mission 3 is not used.
- **Disclosed caveats:**
  - single seed for most results;
  - the pipeline's shape was chosen over logged iterations (final parameters are train-tuned);
  - the availability features use raw timestamps that the published baselines did not have (that
    asymmetry is the R8 contribution itself).

---

## 6. Results so far

### 6.1 Lightweight benchmark: the headline (combined model, cloud run 31 Oct 2026, 2× L40S)

| | Ours | Best published | Earlier phase (ours) |
|---|---|---|---|
| **M1 event-wise F0.5** | **0.866** (P 0.974, R 0.600) | 0.786 (Telemanom-ESA, pruned); SMED 2025: 0.801 | 0.853 |
| **M2 event-wise F0.5** | **0.987** (P 0.985, R 0.997) | 0.949 (Windowed iForest); SMED 2025: 0.971 | 0.964 |

All aspects:

| Aspect | M1 | M2 |
|---|---|---|
| Event-wise F0.5 | 0.866 | 0.987 |
| Channel-aware F0.5 | 0.592 | 0.791 |
| Subsystem-aware F0.5 | 0.600 | 0.997 |
| Alarm precision | 0.886 (ESA best: 0.875) | 0.936 |
| ADTQC | 0.255 | 0.970 |
| Affiliation F0.5 | 0.726 | 0.985 |
| Anomaly-only F0.5 | 0.939 | 0.381 (only 1 anomaly in the M2 light test) |

- M1 test events caught: **25 of 29 anomalies, 14 of 36 rare events.**
- M2: 348 of 349 rare events and the 1 anomaly.
- Result files: `ESA/Main/out 2/M1/result.json`, `ESA/Main/out 2/M2/result.json` (copies of
  `new_runs/results_cloud_2026-10-31/out/`).
- **Weak spot:** ADTQC on M1 (0.255). Alarms often start before the labelled onset.

### 6.2 Full channel set (M1 76 channels, cloud): a negative result, reported honestly

- **M1 full test: F0.5 0.023** (P 0.018, R 0.758). The validation folds had said about 0.87.
- **Cause: false-alarm time.** OR-ing alarms over 58 channels means ESA's `(1 − FPt/Nt)` factor
  collapses precision.
- Likely trigger: STD-k firing for years on near-constant channels. A fix was written
  (`std_floor_fix`) but never run on the full set.
- The pod ran out of credit, and the full-set models, the M2 full result and the configurations
  were lost.
- An earlier full-set score of 0.439 was **not causal** (z-scores used statistics of the whole test
  series), so it must not be claimed as online detection.
- This led to per-subsystem models.

### 6.3 Per-subsystem models (`new_runs/run_sub.py`, `core/subsets.json`)

**Design:**
- 8 whole-subsystem models: M1_S1/S3/S5/S6, M2_S1/S2/S5/S6.
- Each takes up to 8 **context channels** from anywhere in its mission (|corr| ≥ 0.2 on training
  data, near-copies skipped). Context channels are inputs only; each channel is scored by its own
  subsystem model.
- The loss uses target channels only. The STD-k floor fix is on. Telecommands of priority ≥3.
- A model gets its own detector search only if folds A + B hold ≥20 labelled events; otherwise it
  uses its mission's lightweight setting.

| Model | Test events (anomalies) | F0.5 | P / R | Notes |
|---|---|---|---|---|
| M1_S3 (7 targets) | 14 (5) | **0.870** | 0.856 / 0.929 | 13 of 14 caught; silence catches most rare events. Was 0.827 before the MPS bug fix (section 7) |
| M1_S6 (42 targets, Kaggle T4, own search) | 80 (46) | **0.679** | 0.961 / 0.313 | 25 of 80 caught; misses cluster in 2007/08/10/13 (ESA reports drift there). Fixed limits catch 28 but alarm 25% of the time |
| M1_S1 (3 targets) | 5 (3) | 0.600 | 0.599 / 0.600 | only the silence detector catches its events; the forecaster adds nothing |
| M2_S2 (3 targets) | 4 (3) | 0.025 | 0.020 / 0.750 | multi-day alarms during slow regime swings. A genuine weakness |
| M1_S5 | 65 (29) | ? | | was running on Kaggle; result not yet collected. Note: M1_S5 = the M1 lightweight set |
| M2_S1, M2_S5, M2_S6 | 351 / 26 / 26 | **not run** | | need a GPU (Kaggle quota / the RTX PC) |

- M1 combined, partial (S5 missing, so silent): F0.5 0.729 (P 0.938, R 0.385).
- File: `new_runs/out_sub/SUMMARY.md`.

### 6.4 Operator in the loop: monthly retraining with operator feedback (M1 lightweight)

**Question:** after deployment, can the model improve itself from what it sees, with an operator
correcting its alarms?

**Setup (`core/online.py`, `run_online.py`):**

1. The 0.866 model is deployed on 1 January 2007 and monitors 2007–2013 month by month (84 months).
2. At the end of each month, the frozen detector setting turns that month's errors into alarms,
   using only data up to the month end.
3. A **simulated operator** reviews **only the alarms that were raised** (missed events stay
   unseen, as in real operations). Each alarm is classified:
   - overlaps a labelled anomaly or rare event → **confirmed**;
   - starts within 1 h of a priority-3 command → **commanded**;
   - otherwise → **false alarm**.
4. The model is **fine-tuned on everything seen so far** (training years plus deployment months):
   - warm start, learning rate 3e-4, 300 steps;
   - half the batches from the last 90 days, half replayed from all history;
   - rows of confirmed events are **left out of the loss**, so the model does not learn the
     anomaly as normal;
   - false alarms stay in as normal behaviour.
5. **Safety gate:** the update is kept only if it forecasts a held-out last week better, and does
   not get more than 10% worse on the original validation stretch.
6. The updated model monitors the next month, which it has never seen.

**Choosing the fine-tuning setting without the test.** The whole loop was rehearsed inside the
training years: fold B (2005–2006), deploying fold B's model.

| Rehearsal, 2005–2006 | F0.5 | Recall |
|---|---|---|
| Frozen | 0.908 | 0.667 |
| learning rate 1e-4 | 0.931 | 0.733 |
| **learning rate 3e-4 (chosen)** | **0.951** | **0.800** |
| learning rate 1e-3 | 0.951 | 0.800 |

The tie went to 3e-4 because its precision was higher.

**Test 2007–2013, same 84 months:**

| | Frozen | **Retrained + operator feedback** |
|---|---|---|
| **Event-wise F0.5** | 0.866 | **0.888** |
| Precision | 0.974 | **0.998** |
| Recall | 0.600 | **0.615** |
| Channel-aware F0.5 | 0.592 | **0.608** |
| Alarm precision | 0.886 | **0.889** |
| ADTQC | 0.255 | **0.276** |
| Affiliation F0.5 | 0.726 | **0.735** |
| Anomaly-only F0.5 | 0.939 | **0.968** |

**Ablation: does the operator matter?** Same months (Jan 2007 – Oct 2013, 63 events; the
no-feedback run hit the 8-hour cap at month 82):

| | F0.5 | Precision | Recall | Channel F0.5 | ADTQC | Affiliation F0.5 |
|---|---|---|---|---|---|---|
| Frozen | 0.860 | 0.972 | 0.587 | 0.579 | 0.269 | 0.717 |
| Retrained, no feedback | 0.876 | 0.998 | 0.587 | 0.577 | 0.269 | 0.714 |
| **Retrained + operator feedback** | **0.883** | **0.998** | **0.603** | **0.595** | **0.291** | **0.726** |

**Reading:**
- Retraining alone cuts false-alarm time (precision 0.972 → 0.998).
- **Operator feedback** is what adds caught events and improves the channel, timing and
  affiliation scores.

**Cumulative F0.5 by year end** (the "learning over time" curve):

| Year end | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 |
|---|---|---|---|---|---|---|---|
| Frozen | 0.963 | 0.977 | 0.889 | 0.899 | 0.866 | 0.850 | 0.866 |
| No feedback | 0.963 | 0.977 | 0.920 | 0.924 | — | 0.869 | — |
| **With feedback** | 0.963 | 0.977 | **0.931** | **0.932** | **0.895** | **0.877** | **0.888** |

**Integrity checks:**
- The month-by-month alarms equal a single offline pass over the final error stream (0 rows
  differ), so nothing looked ahead.
- 77 of 83 updates were kept by the safety gate.
- Alarm time is the same as frozen (1.05% vs 1.03%).

**Honest caveats:**
- The gain is modest and concentrated in 2009 (0.768 vs 0.576 that year); other years are equal.
- Single seed.
- The simulated operator uses ESA labels; a real operator makes mistakes.
- Telecommand-caused false alarms can only be approximated from timing, because ESA does not
  record the cause.
- This is a **deployment-mode** result, reported separately from the benchmark protocol.

**Operator console (dashboard):** `new_runs/dashboard/` is a static page, localhost only. Serve it
with:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory dashboard
```

It replays the recorded run:
- telemetry strips for channels 41–46;
- the alarm queue with operator verdicts (confirmed anomaly / rare event / false alarm /
  explained by telecommand);
- a telecommand feed;
- the model version and retraining log;
- the live cumulative F0.5 curve (updated vs frozen vs ESA's 0.786);
- a results tab.

`#t=2009-06-01` in the URL jumps to a date. Data comes from `export_dashboard.py`.

### 6.5 Transfer learning: a "new satellite" with little data (M1 lightweight)

**Setup:**
- A new spacecraft has only its **last N days** before deployment (14 or 63 days).
- It trains from scratch, or starts from another model's shared weights; embeddings start fresh.
- Tested once on the full 2007–2013 test with the frozen M1 detector setting.
- Transfer gain = F0.5(transfer) − F0.5(scratch) at the same N.
- Runner: `run_tl.py`; options `NR_TRAIN_DAYS`, `NR_TRAIN_FROM_END=1`, `NR_INIT`, `NR_FREEZE=tcn`.

| Data | Scratch | From M2 (other spacecraft) | From M2, TCN frozen | From M1_S3 (same spacecraft, other subsystem) |
|---|---|---|---|---|
| **14 days** | 0.885 | **0.891 (+0.006)** | 0.872 (−0.013) | 0.879 (−0.006) |
| 63 days | 0.870 | 0.870 (+0.000), 23% less alarm time | 0.853 (−0.017) | 0.854 (−0.016) |

Reference: all 7 training years gives 0.866.

**Reading:**
- With 2 weeks of data, transfer from Mission 2 helps slightly: one more event caught (recall
  0.631 vs 0.615) at the same precision.
- The other sources hurt slightly. That is **negative transfer**: the right source matters more
  than "same spacecraft".
- **Surprise:** 2 weeks scores higher than 9 weeks, and both edge past 7 years. Likely a recency
  effect: the weeks right before deployment resemble the test period best. Worth a sentence, and
  worth seeds before any strong claim.
- Single seed; differences are 1 event.
- Table: `new_runs/out_tl/SUMMARY.md`.

**Note on definitions.** The user's definition of transfer learning is "a replacement satellite of
the same design". ESA has one satellite per mission, so on real data it can only be emulated
(cross-mission, or a short history). The simulator (section 8) gives a true same-design fleet.

### 6.6 Federated learning

- **Earlier prototype** (`ESA/Main/federated/run_federated.py`, older window-GDN model): FedProx
  μ = 0.01, trimmed-mean aggregation, personalisation fine-tune, M1 and M2 as clients.
  - Results: M2 0.923 → 0.936, but M1 0.080 → 0.018 (worse).
  - Hypothesis: the two lightweight sets are mismatched subsystems (M1 = S5, M2 = part of S1).
- **On the current model: not run yet.** Planned for the Basilisk fleet (section 8) and,
  optionally, S5 ↔ S5 / S6 ↔ S6 on ESA data.

### 6.7 Basilisk simulator: installed and smoke-tested

- Basilisk 2.12.2 in its own conda env `bsk` (Python 3.11; `pytest` and `matplotlib` added by
  hand).
- Vizard 2.4 (the 3D viewer) is installed on the Mac.
- Sample scenario `ESA/Main/sim/sample_rw_scenario.py`:
  - 750 kg spacecraft, 4 Honeywell HR16 reaction wheels (Basilisk's preset, pyramid layout);
  - LEO at about 500 km;
  - inertial pointing → MRP feedback control → wheel torques;
  - at minute 10 a "telecommand" changes the target and the spacecraft slews;
  - final attitude error 2.7e-6.
- Outputs:
  - `ESA/Main/sim/_VizFiles/sample_rw_scenario_UnityViz.bin` (open in Vizard);
  - `sample_telemetry.png` (attitude error, wheel speeds, wheel torques).

---

## 7. Bugs found and fixed (worth knowing)

1. **Apple GPU (MPS) asynchronous copy bug (2 Nov 2026).**
   - Cause: `core/train.py` `to()` used `non_blocking=True`. On MPS, the command-feature copy in
     the error pass came from a temporary buffer that was freed too early.
   - Effect: NaN and garbage forecast errors (up to 2e37) → long false alarms.
   - Fix: `non_blocking` only on CUDA. Online/TL code now asserts that errors are finite.
   - Laptop subsystem results were re-scored with `rescore_sub.py`. M1_S3 0.827 → 0.870;
     M1_S1 unchanged; M2_S2 0.036 → 0.025.
   - Cloud and Kaggle (CUDA) results were never affected.
2. **STD-k on near-constant channels** (full set): years-long alarms. Fix: `std_floor_fix`.
3. **Non-causal z-scores** in the old full-set run (0.439): must not be claimed as online.
4. **Earlier fixes, for the record:**
   - float16 overflow in error files (clipped at 6e4);
   - label mask crashing on labelled context channels;
   - the scorer erroring on folds without events;
   - the evaluation waiting too long for the fixed-limit baselines;
   - the RAM-sized search profiles for big subsystems.

---

## 8. The next phase: onboard + federated + simulated fleet

### 8.1 Architecture

**On each satellite (onboard):**
- int8 detector (~0.2 MB) runs on every new telemetry row;
- an alarm triggers a tiered FDIR response;
- only alarms plus compact summaries are sent down.

**Between satellites (inter-satellite links):**
- each satellite sends model updates (shared layers only; embeddings stay local);
- a rotating leader aggregates them with robust filtering (trimmed mean), so a faulty satellite
  cannot poison the fleet model;
- the aggregated model is broadcast back, and each satellite fine-tunes it locally
  (personalisation).

**Ground:**
- operators review alarms (confirm / reject / commanded);
- verdicts are uplinked and used as loss masks in the next local update;
- the ground is not the aggregator.

### 8.2 Simulated fleet (Basilisk)

- **3 satellites**, same design (based on the official reaction-wheel scenario, HR16 wheels), with
  small per-satellite variations: sensor noise, wheel friction, inertia, orbit phase.
  Optionally, orbits from real published TLEs.
- **Telemetry, about 15–20 channels at 30 s:**
  - 4 wheel speeds, 4 wheel currents/torques;
  - attitude error (3), body rates/gyro (3);
  - star-tracker status;
  - battery charge, solar power, bus load, eclipse flag;
  - realistic noise and telemetry dropouts (ground-contact gaps), so the availability features and
    the silence detector behave as on ESA data.
- **Telecommands:** scripted slews, momentum dumps, mode changes.
- **Faults and degradation, with exact labels:**
  - wheel friction rising (lubrication decay, the leading real failure cause);
  - wheel stuck or failed;
  - gyro bias drift;
  - stuck sensor (frozen value);
  - telemetry dropout;
  - battery capacity fade.
- **Export in exact ESA-ADB format** (channels.csv, labels.csv, anomaly_types.csv, telecommands),
  so the existing pipeline runs unchanged.

**Why one satellite cannot do it alone (the honest version).** Our model learns **normal
behaviour**, not fault signatures, so "satellite 1 never saw a gyro fault, therefore misses it" is
not the mechanism. The real limit is each satellite's **narrow view of normal**:

| | Satellite 1 | Satellite 2 | Satellite 3 |
|---|---|---|---|
| Normal operations it trained on | sun-pointing only | frequent slews | eclipse seasons + momentum dumps |
| Fault it experienced | wheel friction rise | gyro bias drift | battery fade |

At test time every satellite meets all modes and all faults.
- **Alone,** a satellite raises false alarms on normal behaviour it never saw, and misses faults
  hidden inside it.
- **Federated,** it has learned the others' normal behaviour without seeing their data.
- **Plus:** federated detector-setting selection lets fault labels on one satellite improve
  thresholds on all.

### 8.3 Experiments and metrics (fixed before running)

| Experiment | Arms | Metrics |
|---|---|---|
| Fleet federated learning | solo · federated · federated + personalisation · centralised (upper bound) | ESA F0.5 / P / R per satellite; false alarms per day on unseen normal modes; **recall on fault types first seen on another satellite** |
| Federated operator feedback | operator verdict on satellite 1 → next round on satellites 2–3 | detection delay on 2–3 before vs after |
| Robust aggregation | one satellite's updates corrupted | fleet F0.5 with plain averaging vs trimmed mean |
| Degradation early warning | wheel friction ramps | lead time before a fixed threshold trips; time-to-failure estimate (trend on a health indicator) |
| Transfer to a replacement satellite | satellite 4 launches with 2 weeks of data: scratch vs fleet model | transfer gain |
| Closed-loop FDIR | detector alarm → isolate wheel → reconfigure to 3 wheels | pointing error before and after; time from fault to recovery |
| Onboard feasibility | fp32 vs fp16 vs int8 on the ESA test (same detector setting) | all ESA aspects; model size; time per 30 s step on a Cortex-A53 (Raspberry Pi) or ARM CPU |

### 8.4 Plan (a few days)

1. **Simulator realism:** satellite, telecommands, faults and degradation, 3-satellite fleet,
   ESA-format export, sanity checks:
   - momentum conserved when no torque acts;
   - wheel speeds within HR16 limits;
   - control settles;
   - nominal runs alarm-free.
2. Detection and isolation on the fleet; federated vs solo; transfer to a new satellite.
3. Degradation early warning and federated operator feedback.
4. Closed-loop tiered FDIR; int8 quantization and edge timing; dashboard view; write-up.

If time runs out, steps 1–2 already make a complete story (validated model + fleet federated
learning).

### 8.5 Scope check: what was cut, and why

| Original claim | Verdict |
|---|---|
| Full system as NASA cFS flight software | Future work. Porting TCN + GATv2 + Telemanom + histograms to C is weeks |
| Running on LEON3 via emulation | Future work. Needs the RTEMS toolchain, a C port, and a cycle-accurate simulator (Gaisler TSIM); QEMU timing is not cycle-accurate |
| Onboard **training** for federated learning | Must be measured or limited (e.g. only the personalised layers). A month's fine-tune takes ~100 s on the Mac GPU; on a Cortex-A53 CPU, likely tens of minutes to hours |
| "No ground involvement" | Rephrased: the ground is not the aggregator, but operators still review alarms. The operator result depends on it |
| "Degradation on a few spacecraft becomes early warning for all" | A hypothesis to test, not a promise |

**Simulator choice.**
- Basilisk is the best **available** option: open, peer-reviewed, real hardware presets,
  scriptable, fast enough for fleets.
- More realistic options exist but are out of reach:
  - agency/manufacturer mission simulators (ESA SIMULUS/SIMSAT-based, Airbus, Thales) are not
    public;
  - NASA NOS3 runs real flight software, but is heavy and not built for fleets → future work.
- Atmospheric, optical and communication simulators (G-SDSU, HYGEOS, OS³, SatScope) solve
  different problems and do not fit.
- Simulated results are always reported **separately** from ESA results.

---

## 9. Where everything is

| What | Path |
|---|---|
| Main project root | `/Users/Shreyas/Downloads/Capstone/ESA/Main/` |
| Original project notes (data facts, environment landmines, rules) | `ESA/Main/CLAUDE.md` |
| Current pipeline (model, detectors, scoring, all runners) | `ESA/Main/new_runs/` (`core/`, `run_*.py`) |
| Lightweight models and results (headline) | `ESA/Main/out 2/M1`, `ESA/Main/out 2/M2` (model, frozen.json, result.json, events, alarms) |
| Per-subsystem outputs | `ESA/Main/new_runs/out_sub/<model>/`, `out_sub/SUMMARY.md` |
| Operator-loop outputs | `ESA/Main/new_runs/out_online/` (`test/alarm__lr=0.0003/`, `test/nofb__lr=0.0003/`, `foldB/…`, `base/`, `chosen.json`, logs) |
| Transfer-learning outputs | `ESA/Main/new_runs/out_tl/` (`SUMMARY.md`, one folder per run) |
| Dashboard | `ESA/Main/new_runs/dashboard/` (+ `export_dashboard.py`) |
| Kaggle notebook for subsystems | `ESA/Main/kaggle_sub/run_subsystem.ipynb` (+ `new_runs_sub_code.tar.gz`, `new_runs_full_data.tar.gz`) |
| Earlier federated prototype | `ESA/Main/federated/run_federated.py` |
| Basilisk sample | `ESA/Main/sim/` |
| ESA raw data | `ESA/ESA-Mission1/`, `ESA/ESA-Mission2/`, `ESA/ESA-ADB/` |
| GitHub | `shreyaskodithyala2/Capstone`, branch `new` (datasets are never committed) |

**Environments:**
- `timeeval` (Python 3.9, numpy 1.21.6, torch with MPS) for the ESA pipeline;
- `bsk` (Python 3.11, Basilisk 2.12.2) for the simulator.

**Machines:**
- **MacBook (M-series, 16 GB, ~35 GB free disk):** keep it responsive.
- **RTX PC** (i9-14900, 32 threads, 128 GB RAM, 1 TB, RTX 4500 Ada 24 GB, Ubuntu 24.04): for heavy
  runs. Access via SSH (VS Code Remote-SSH, Tailscale if off-network); AnyDesk only for viewing.
- No cloud budget left.

---

## 10. Open items and risks

1. **Seeds.** Every headline is single-seed; 3 seeds are needed before any paper.
2. **Missing subsystem runs:** M1_S5 (Kaggle result not collected), M2_S1, M2_S5, M2_S6.
3. **Federated learning on the current model:** not yet run.
4. **ADTQC on M1 is low (0.255):** alarms tend to start early.
5. **M2_S2 false-alarm time** in slow regime swings: unsolved.
6. **Simulator realism:** Basilisk data is cleaner than real telemetry. Add noise, dropouts and
   spread, and never mix simulator numbers with benchmark numbers.
7. **Novelty claims:** verify with a proper literature search before writing "first".
