# HANDOFF: Basilisk fleet v2, federated learning and RUL (living log)

Read first: `CLAUDE.md` (rules and context), then this file. It records **what was done, why, where
it is, and what is next**, so work can resume from here. The newest status is at the top of §0.

## 0. Status (newest first)

**Full handover, 5 Nov 2026.** `FULL_PROJECT_REPORT.md` is now the complete report and `MANIFEST.md` says where every file is. Use them over older counts in sections 1-2 of this file.
- **Storage:** code and small results go in git. `data/`, `fed/out_final`, `fed/out_v2` and `fed/out_w60` were packed into 8 pieces (15.7 GB, 1,176 files) for GitHub Release `v1.0-data`. All 8 checksums passed, and every file read back from the pieces matched the original byte for byte (`results/audit/verify_release.log`).
- **Added to git-tracked folders:** `results/dataset_meta/` (labels and truth), `results/logs/`, `results/audit/scratch/`, more final outputs in `results/final/`, generated `all_metrics.md` and `per_event_detection.csv`, and exact package lists in `envs/localenv_*`. No models were retrained and no scores changed.
- **Newly highlighted from the partial audit log:** shifting all alarms 70 rows later lowers scores sharply (for example FL + personalization on SAT_1, .892 to .357 on all events). This is a diagnostic, but it shows the scores depend on alarm timing.
- **Still stale:** `CLAUDE.md` and `REPORT.md` still say "beats every published result". The user has not asked for them to be edited.
- **User decisions still open:** report section 18. Nothing has been committed or pushed by Claude; the user does that.

**Causality audit, 4 Nov 2026, stopped early at the user's request (`results/audit/audit_causality.py`, partial log `results/audit/causality_full.txt`).** Finished: 14,500 training batches replayed, last row day 116.45; all 12 models show zero gradient into the target row or later, and altering or deleting the future changes earlier errors only by rounding noise (under 1e-5); 54 cuts inside alarms show no differences beyond the 70-row limit. Open: 3,428 of about 84M resampled values on SAT_1 and SAT_2 differ from the raw check, cause not investigated. Not finished: scrambled-label test, RUL cut rerun, zero-look-ahead scores for FL + personalization fleet on SAT_2-4. Partial zero-look-ahead results show scores depend on the allowed 35-minute look-ahead. RUL lead is 99 days, not 100, because day 256 is complete only at its end.

**Matched-settings view, 4 Nov 2026 (`fed/out_final/matched_*.json`):** with the final fleet settings for all models, means (all / anomalies) are solo .818/.627, plain FL .783/.587, FL + personalization .902/.773. FL + personalization beats solo under all three fleet settings, but the settings carry much of the gain (SAT_2: .670 own, .864 same solo model with final settings, .923 FL + personalization). Mechanism unproven: validation loss after personalization beats solo on SAT_1 and SAT_3 only. Solo saw less training than FL + personalization; no matched-budget arm.

**Failing-satellite question, 4 Nov 2026: no corruption test exists.** FL used SAT_2 days 0-137 only (gradients to day 116, round selection to day 137); detector settings used days 137-182. Wear starts day 190 and failure is day 356.1, both in the frozen test period, so the failure could not reach FL. This is timing, not a defence: the aggregate is a plain mean. No per-round F0.5 was logged (only validation loss, `results/final/fed_history.json`); personalisation is one local stage after FL, not rounds. Proposed, not started: stress run with SAT_2 wear days in FL, with and without a health-based guard.

**Whole-project review, 4 Nov 2026:** see `results/project_synthesis_review.md`. Main points:
- Goetze et al. (IEEE SCC 2025) report M1 0.927, very likely on the same 65-event test, so "beats every published result" no longer holds on M1. `CLAUDE.md` and `REPORT.md` not edited.
- "SMED 2025" could not be found. Find the citation or remove it.
- Fleet gain by step: settings sharing +.196, plain FL weights +.008, fine-tune +.176 (point estimates; means .393, .589, .597, .773). Plain FL forecasts worse than solo on all four satellites. Recall on locally unseen faults rises 13 to 18 of 27.
- The headline fleet arm uses fleet-selected settings, which conflicts with the per-satellite rule in `CLAUDE.md` section 5. The compliant result is .393 to .485.
- Atefrad & Karami read in full: window-level metrics, slices of one satellite as clients, numbers that do not reconcile. Section 9 of the review has the rule-by-rule comparison and the stand-out ideas. `pymupdf` added to the bsk env inside `.localenv` to read PDFs.


- **4 Nov 2026, paper-readiness review done: `results/paper_readiness_review.md`, evidence in `results/audit/`.** Integrity checks pass. Not top-tier ready: single seed and
  one fleet; RUL is partly circular (same failure level, same exponential, same wheel); sensors are open loop; 73 % of anomalies pass the one-liner test.
  **Corrections to earlier text:** ESA anomalies are mostly multivariate global, not local (§1, §9.3 wording is wrong). "Beats every published result" in `CLAUDE.md` is false as stated:
  anomalies-only M1 is 0.939 vs ESA's 0.968, GDN2 used telecommands that ESA's lightweight subsets exclude, and Goetze et al. 2026 report 92.7 on M1 lightweight.

- **4 Nov 2026, 16:45: FINAL fresh-seed fleet scored once (`sim/fleet_final.json`).** Full table in §5 "Final result". Results copied to
  `results/` because `fed/out_*` is gitignored. Waiting for the user: is this good enough, and which arm is the headline?
  Do not iterate on this fleet: it has been scored, so any further change needs another fresh fleet.

- **4 Nov 2026, M5 Max session (borrowed machine: 18 cores = 6P + 12E, 64 GB).**
  - **Hard rule on this machine:** it is not the user's. Everything stays inside `capstone_pr/` so the
    folder can simply be deleted. No installs outside it, no access outside it, no git account or
    config changes, no commits or pushes (the user commits at the end).
  - **Toolchain is in `.localenv/` (gitignored).** Run `source .localenv/activate.sh` before every
    command: it points `HOME`, temp and all caches into the folder and puts the in-folder conda first.
  - **Mistake made and undone:** Miniforge was first installed with Homebrew, outside the folder, before
    the user stated this rule. Removed: the cask, the `conda` / `mamba` symlinks, `~/.conda`, the cached
    installer. Not restorable: brew's automatic cleanup deleted two empty log folders
    (`~/Library/Logs/Homebrew/openssl@3`, `.../ca-certificates`) and refreshed its `.cleaned` stamp.
  - **Env change:** `scipy=1.11.4` cannot be solved with `numpy=1.21.6` on conda-forge, so scipy,
    scikit-learn, matplotlib and statsmodels are unpinned (got scipy 1.11.2, sklearn 1.6.1, statsmodels 0.14.2).
  - **Speed changes (results unchanged):**
    - twin follow mode (`physics.py sync`, `run_fleet.py`): twins run 2 days behind the faulty run instead
      of after it. Tested bit-identical to replay on 6 days with 74 forced dumps. `--no-follow` restores
      the old order. `code_sha256` therefore differs from `fleet_v2b.json`; the warning is expected.
    - `fedsim.py`: one process per satellite (solo, pers, errors), per arm (detect), and a worker per
      satellite inside each FL round. Embedding init is now seeded per satellite (`1 + index`).
  - **Reporting:** `fed/report_round.py` prints every metric. New per-arm fields: false-alarm events, per
    30 days, event share, time share, rare events caught. Results are now `results_<arm>.json`, merged
    into `results.json`; delete `results*.json` to redo detection.

- **4 Nov 2026, ~13:00: moving to another machine** (M5 Max or RTX PC).
  - Round 2 was mid-run on the old Mac:
    - solo SAT_1–3 done (best val 0.054 / 0.068 / 0.035), SAT_4 training;
    - then fed, pers, errors and detection (~2.5 h more).
  - **On the new machine, rerun round 2 from scratch** (§6). It is deterministic from `sim/fleet_v2b.json`.
  - If the old Mac finishes first, its `fed/out_v2/results.json` (small) can be copied over for comparison.
  - **Next after round 2:** report solo vs fed scores, unseen-family recall and RUL to the user, then ask.
    - If the results are clearly bad, apply the easiest realistic lever (§8) and rerun.
    - The user allows making the dataset slightly easier, keeping it ESA-like.

- **4 Nov 2026: round 1 failed on precision, found to be a simulator realism bug; round 2 is regenerating.**
  - Round-1 outputs are in `fed/out_v2_r1/`.
  - **Fix:** profiled slews (`operations.Timeline.slew`, the `slew_pt` event in `physics.py`) and agile
    controller gains (K 30, P 150).
  - Fresh seeds; `data/gen_v2.log`.
  - Next: validate, baselines, `fed/fedsim.py`, `fed/rul.py`, then report to the user.

- **4 Nov 2026: v2 dataset generated and validated.**
  - Generation took 70 min (`data/gen_v2.log`). RUL round 1 done.
  - `fed/fedsim.py` is running (log: `fed/out_v2/log.txt`, `fed/out_v2/run.out`).
  - Epoch time is 138 s on MPS, so the full pipeline takes about 5 h.

- **4 Nov 2026: v2 dataset generation started.**
  - Command: `sim/run_fleet.py`; log in `data/gen_v2.log`.
  - Next steps: validate, then `fed/fedsim.py` (solo → fed → pers → errors → detect), then `fed/rul.py`.
- **v1 dataset (5 satellites, 34 channels)** kept as `data/sim_fleet_v1/` (validation reports inside).
  - Superseded because it was too easy for limit checks: rare events dominated.
  - The v1 working files (`data/sim_work`) were deleted, which was needed for disk space (≈8 GB free).

## 1. Goal of this stage (agreed with the user, 4 Nov 2026)

1. **Dataset:** a realistic, ESA-ADB-format, 4-satellite fleet. It must be hard enough not to look trivial,
   but scorable.
2. **Detector:** the same GDN2 detector as ESA-ADB, run on each satellite alone (solo). Scores should be
   decent but not perfect.
3. **Federated learning (FL) with personalisation** must improve the scores. Above all, it must improve
   detection of **fault families a satellite never saw in its own training data**.
4. **RUL:** a wheel degrades, becomes visible in about the last 2 months, and fails near the end. We
   must predict this early, with a dynamic threshold. The user said "keep it simple".
5. **Reporting:** after each round, report per-satellite solo F0.5, FL F0.5 and the RUL lead time. Then
   ask the user whether to iterate, unless the results are clearly bad, in which case fix them first.
6. **Decided:** no fault-type classifier for now. Family attribution is rule-based, from the alarmed
   channels. Labels are complete, so a classifier can be added later without regenerating.

## 2. Dataset v2 design (`sim/config.py`, `sim/operations.py`)

- **Fleet:** 4 satellites, 365 days, epoch 2025-01-01.
  - Train = days 0–181; validation = the last 45 days of train (days 137–181); test = days 182–364.
  - Archive ARC_1..3 are 150-day run-to-failure wheels, used only for the RUL failure threshold.
- **Channels (20):**
  - rw1–4 speed, rw1–4 current, gyro x/y/z, att_err (total);
  - mag x/y/z, sa_power, bat_voltage, bus_load, bat_temp;
  - aocs_mode (context, not a target).
- **Telecommands:** 12, all priority 3.
- **Fault families:**
  - F1 wheel: A1 friction spike, A2 drive dropout, W1–W3 wear (SAT_2).
  - F2 sensor: B1 gyro bias drift, B2 gyro spikes (point), B4 star-tracker outage.
  - F3 power/thermal: C1 solar string open, C3 unit latch-up, D1 heater stuck on.
- **Who sees what** (the case for federated learning: every family is seen in training somewhere, and
  every satellite meets new families in test):

| Sat | Role / orbit | Train faults | Test faults | Notes |
|---|---|---|---|---|
| SAT_1 healthy | survey, SSO 10:30 | F2 (4) | F2, F1, F3 (isolated) | few anomalies |
| SAT_2 wheel | imaging, 65° | F1 (10) | F1, F2 (some overlapping F1), wear spikes, degradation, failure | wheel rw2 wears; visible ≈ day 300; fails ≈ day 354 |
| SAT_3 power | survey, 53° | F3 (10) | F3, F2, F1 | |
| SAT_4 ops | imaging (heavy), SSO 13:30 | F1 + F2 (5) | F3 | ~50 rare events; about 2× the telecommands |

- **Rare nominal events:**
  - Sats 1–3: quarterly gyro and inertial calibrations, 0.5 star-tracker resets per month, 1 safe drill.
  - SAT_4: monthly calibrations, 2 resets per month, 4 drills.
- **Fault placement:**
  - The first fault of each trained type goes inside the validation window, so detector settings can be
    chosen there.
  - The rest are spread evenly over time, with random offsets.
- **Slews (v2b):** profiled minimum-jerk guidance with rate feed-forward (`Timeline.slew`, `slew_pt`),
  120–240 s long. MRP gains K 30, P 150. Calibration and safe-mode steps stay steps; they are labelled
  rare events.
- **Slow ageing** (`AGEING`): solar cells −2.75 %/yr, battery capacity −2 %/yr, internal resistance +5 %/yr.
- **Labels:**
  - Computed against a fault-free twin run with shared noise (4σ rule), as in v1.
  - `truth/events_truth.csv` now has a `FaultFamily` column (F1/F2/F3, R = rare event).

## 3. Experiment design (fixed before training; `fed/fedsim.py`)

- **Model:** GDN2 exactly as on ESA-ADB (`ESA/Main/new_runs/core`, imported unchanged), seed 0.
  - Labelled training rows are left out of the loss.
  - It trains only on rows before the validation window.
- **Arms (same number of local gradient steps per satellite: 12 × 150 batches):**
  - **solo:** a separate model per satellite.
  - **fed:** FedAvg of `shared_state()`, i.e. everything except the channel and command embeddings,
    which stay local.
    - Rounds are led by a rotating leader; equal weights.
    - Early stopping on the mean validation loss.
  - **pers:** fed plus a local fine-tune of all weights (100 batches, lr 3e-4), keeping the best own
    validation loss.
- **Detector settings:** the ESA detection stack (`core/period.py`: past-only robust z, two-tier,
  STD-k, drift, close and minimum length).
  - Coordinate search starting from the ESA M2 frozen setting.
  - **Local selection:** each satellite maximises its own validation F0.5.
  - **Federated selection:** the fleet maximises the mean validation F0.5.
    - Only one score per candidate setting is exchanged (federated hyperparameter tuning).
    - Each satellite applies the setting to its own error scale.
  - Telemanom pruning is not used here (cost); report that.
- **Test half scored once, per satellite:**
  - ESA event-wise F0.5 (Anomaly + Rare Event, and Anomaly only), channel-aware F0.5, alarming
    precision, ADTQC;
  - recall per family, recall on families **unseen locally**, detection count, median delay;
  - family attribution from the alarmed channels.
- **Headline comparison:** solo + local selection vs pers + federated selection. All arm × selection
  combinations are reported (ablation).
- **Not run (cost):** centralised upper bound, multiple seeds.

## 4. RUL (`fed/rul.py`)

- **Health indicator:** daily median |current| of a wheel ÷ the median of the other three. It is a
  z-score against the healthy reference period (days 0–136), smoothed with a 7-day median. No labels.
- **Thresholds:**
  - alert: mean + 6 sd of the reference period, 3 days running;
  - failure: the lowest HI any archive wheel reached on its last day before failing.
- **RUL:** log-linear fit (exponential wear) over the last 30 days, extrapolated to the failure threshold.
  The baseline is a straight-line fit.
- **Metrics:** alert lead time, RMSE/MAE over the last 60 days, PHM08 score, α-λ (±20 %).
- **Figure:** `fed/out_v2/rul_SAT_2.png`.

## 5. Rounds log (results; every round kept, including losing ones)

### Dataset v2 facts (generated 4 Nov 2026; seeds in `data/sim_fleet/fleet.json`)

| Sat | Train: anomalies / rare events | Test: anomalies / rare events | Test families | Anomalous rows (train / test) |
|---|---|---|---|---|
| SAT_1 | 4 (F2) / 5 | 6 / 4 | F1 2, F2 1, F3 3 | 0.07 % / 0.85 % |
| SAT_2 | 10 (F1) / 4 | 17 / 6 | F1 9, F2 8 | 0.27 % / 39 % (one long degradation event, days 294–353, then failure) |
| SAT_3 | 10 (F3) / 4 | 14 / 7 | F3 6, F1 4, F2 4 | 2.2 % / 1.6 % |
| SAT_4 | 5 (F1+F2) / 14 | 5 / 14 | F3 5 | 0.08 % / 1.0 % |

- **Character:**
  - Most anomalies are multivariate-local or univariate-local (within the normal range), as in ESA.
  - Almost no point events: gyro spikes merge into segments.
  - Star-tracker resets (180 s) are invisible (below 4σ), so they are not labelled.
- **Validation:**
  - **Physics:**
    - momentum drift 6e-11;
    - gyro orbit rate 0.061–0.065 vs analytic 0.0627 deg/s (gyro bias);
    - eclipse within 1.3–1.5 min of the analytic cylindrical model.
  - **Similarity to ESA (catch22):** median feature coverage 0.99; nearest-neighbour closeness 0.60
    (v1: 0.64).
  - Output: `data/sim_fleet/validation/validation_part1.json`.
- **Difficulty, out-of-limit check** (`validation/baselines.json`):

| Sat | F0.5, all events | F0.5, anomalies only | Anomalies: precision | Anomalies: recall |
|---|---|---|---|---|
| SAT_1 | 0.83 | 0.50 | 1.0 | 0.17 |
| SAT_2 | 0.77 | 0.66 | 0.96 | 0.29 |
| SAT_3 | 0.66 | 0.39 | 0.67 | 0.14 |
| SAT_4 | 0.90 | 0.39 | 0.50 | 0.20 |

  - The all-events score is high because calibrations (rare events) go out of range.
  - On anomalies, limits catch only 14–29 %.
  - GlobalSTD3/5 and iForest score ≤ 0.034.

### RUL round 1 (`fed/rul.py`, `fed/out_v2/rul.json`, `rul_SAT_2.png`)

- **First try:** the HI was a z-score, so the failure threshold mixed scales (the archive reference is
  5 days, SAT_2's is 137 days).
  - The exponential fit was always early, by about 17 days (MAE 17, α-λ 0).
  - **Logic fix:** the HI is now the excess current fraction (physical, comparable across wheels). Alerting
    still uses z > 6 for 3 days. The threshold rule is unchanged (lowest archive value).
- **After the fix:**
  - Failure threshold 62.4 (archive wheels 71.6 / 62.4 / 65.6).
  - **Alert on day 263, on the correct wheel (rw2); failure on day 353.4, so a 90-day lead time.** The
    true visible degradation starts on day 294.
  - Exponential RUL over the last 60 days: **MAE 3.3 days**, RMSE 5.1, PHM08 20.6, **α-λ(20 %) 0.98**,
    15 % late.
  - Straight-line baseline: MAE 311 days, α-λ 0, always late. This shows the exponential model is needed.


### Round 1 (dataset v2a, step slews): FAILED on precision → simulator fix

Test-half event-wise F0.5 (Anomaly + Rare Event), **solo / fed / pers**, each with local or federated
selection:

| Sat | solo, local selection | fed, federated selection | pers, local selection | pers, federated selection | Recall on unseen families (solo) |
|---|---|---|---|---|---|
| SAT_1 | 0.096 | 0.096 | 0.096 | 0.093 | 0.6 |
| SAT_2 | 0.040 | 0.038 | 0.038 | 0.037 | 0.75 |
| SAT_3 | 0.116 | 0.089 | 0.092 | 0.118 | 0.75 |
| SAT_4 | 0.017 | 0.016 | 0.012 | 0.016 | 0.8 |

- **What we saw:**
  - Recall was fine: 57–95 % of events, including families never seen locally.
  - Alarming precision was 0.8–1.0, but alarms covered 2.5–22 % of nominal test time, so ESA's
    event-wise precision was 0.01–0.10.
- **Diagnosis** (scratch scripts `diag*.py`, results only here):
  - On SAT_4, 81 % of false alarms came within 20 min of a SLEW_TARGET command.
  - SAT_3's false-alarm runs (0.51 per day, median 94 min long) matched its slew rate (0.5 per day).
  - **Cause:** the simulator commanded slews as a **step** of the attitude reference (8–30°). The
    attitude error jumped, the wheels saturated, and the transient's size depended on an unknown angle.
    Real AOCS uses profiled guidance.
- **Fix:**
  - A minimum-jerk slew profile with rate feed-forward through `eulerRotation.angleRates` (this module
    integrates the rates and accepts new `angleSet` values; tested). Updated every 10 s; duration
    120–240 s.
  - MRP gains raised from K 3.5 / P 30 to K 30 / P 150 (ωn ≈ 0.09–0.11 rad/s, ζ ≈ 1).
  - 2-day test on SAT_4: tracking error during slews 0.48° median, 0.85° max; outside slews 0.005°
    (p99); body rate ≤ 0.26°/s; |u| ≤ 0.12 Nm.
  - The remaining slew error comes from having no acceleration feed-forward (α/ωn²).
- **Unchanged:** the fault schedule, channels and detection code. RUL round 1 is valid only for dataset v2a,
  so it will be rerun.

### Round 2 dataset (v2b, profiled slews; regenerated 4 Nov 2026, fresh seeds)

- **Validation:**
  - momentum drift 2e-10;
  - catch22: median coverage 0.99; **nearest-neighbour closeness 0.74** (was 0.60).
- **Out-of-limit check, F0.5 on all events:** SAT_1 0.64, SAT_2 0.55, SAT_3 0.35, SAT_4 0.94. SAT_4 is
  high because of its many calibration rare events.
- **Out-of-limit check, anomalies only:**

| Sat | F0.5 | Precision | Recall |
|---|---|---|---|
| SAT_1 | 0.23 | 0.25 | 0.17 |
| SAT_2 | 0.35 | 0.96 | 0.10 |
| SAT_3 | 0.13 | 0.13 | 0.13 |
| SAT_4 | 0.56 | 1.0 | 0.20 |

- GlobalSTD and iForest ≤ 0.041.
- **RUL v2b:**
  - alert on day 244 on rw2 (correct), i.e. a **112-day lead**;
  - exponential fit over the last 60 days: MAE 5.9 days, RMSE 6.4, PHM08 35.7, α-λ(20 %) 0.39, 3 % late
    (it predicts slightly early).
- **Gotcha:** `run_round.sh` needs `data/sim_fleet/validation/` to exist before it starts. Validate and
  baselines were rerun by hand.

### Round 2 (dataset v2b regenerated on the M5 Max, GDN2 solo/fed/pers): BAD, diagnosed

- **Generation 24.5 min, round 46 min.** v2b reproduced: catch22 0.74, momentum 2e-10, baselines and RUL identical.
- Report: `fed/out_v2/reports/round2.txt`, `round2_results.json`.
- Anomaly F0.5, solo local / pers federated: SAT_1 0.044 / 0.040, SAT_2 0.300 / 0.353, SAT_3 0.279 / 0.063,
  SAT_4 0.556 / 0.556. Caught 4/6, 4/30, 6/15, 1/5 solo.
- **Diagnosis** (`.localenv/tmp/diag.py`, dev-fleet test half):
  - 83-100 % of false alarms came within 20 min of a command, almost all `SLEW_TARGET`, median 0-2 min
    after it. The model sees that a slew was commanded, not its angle, so the transient cannot be
    forecast. Thresholds were pushed to a120 to clear it.
  - Most faults peak at z 30-150; a few are subtle (C1 7-15, B4 3-17, some A2).
  - Selection was dominated by loud calibrations.
- FedAvg forecasts worse than solo at equal steps: mean val 0.0615 vs 0.046. Pers recovers (SAT_4 0.0404 vs 0.0369).
- **RUL:** threshold rule changed from min to max of the archive wheels, chosen by leave-one-out PHM08 on the
  archive only (0.152 vs 0.227). MAE 5.9 -> 3.5 days, alpha-lambda 0.39 -> 0.58, late 3 % -> 15 %. Alert day
  244, 112 days ahead, correct wheel. Reliable within +-20 % only at the very end.

### Round 3 (detector only, no retraining): command mask + anomaly-weighted selection

- Added: `mask` (all channels, after SLEW_TARGET / POINT_NADIR), `smask` (power channels, after payload / TX
  switching), objective = mean of F0.5 all and F0.5 anomalies, per-group thresholds (`_g`), same-setting table.
- Reports `round3.txt` (3a, one mask), `round3b.txt` (3b, separate smask). Per-group thresholds were unstable and lost.
- Anomaly F0.5 / caught, 3b:

| Sat | solo local | fed federated | pers federated |
|---|---|---|---|
| SAT_1 | 0.554, 2/6 | 0.500, 1/6 | 0.213, 3/6 |
| SAT_2 | 0.300, 4/30 | 0.558, 8/30 | 0.328, 23/30 |
| SAT_3 | 0.712, 5/15 | 0.769, 6/15 | 0.420, 8/15 |
| SAT_4 | 0.556, 1/5 | 0.555, 1/5 | 0.555, 1/5 |

- Same-setting comparison is mixed. At a20 m30 s5 pers >= solo on all four (F0.5 +0.011, +0.132, +0.136, 0; false
  alarms 13 vs 14, 53 vs 105, 12 vs 20, 0 vs 0). At a50 m10 s10 solo beats pers on SAT_1 and SAT_2. Fed alone is
  poor at a20 (567 false alarms on SAT_3).
- Remaining false alarms (`.localenv/tmp/diag2.py`, pers at a20): SAT_2 43 of 45 on bat_temp, nearly all in
  days 182-242 (season unseen in training); SAT_1 mostly early test, 62 % near eclipse edges; SAT_3 all att_err,
  all within 6 h after a calibration.

### Round 4 (detector only): per-channel calibrated thresholds, latching, fleet ceilings

- `thr="k<x>"`: x times each channel's own noise ceiling (99.99th percentile of z on nominal validation rows, command-masked).
  `close` up to 720 rows, `gclose` latches across channels, `ceil="fleet"` takes the max ceiling over satellites (19 numbers each).
- Per-group thresholds removed (round 3 loser). Report `round4.txt`.
- Anomaly F0.5, solo local / fed federated / pers federated: SAT_1 0.141 / 0.438 / 0.333; SAT_2 0.308 / 0.696 / 0.856;
  SAT_3 0.484 / 0.847 / 0.769; SAT_4 0.556 / 0.555 / 0.555. Caught 30 -> 39 (fed) of 56. False alarms 82 -> 10.

### Round 5 (dataset v2c: same physics, stronger faint faults; models from round 2, errors and detection redone)

- **Dataset change** (products only, RNG stream unchanged): C1 2-3 strings (was 1-2), C3 +90-160 W (was 50-90), B4 estimator
  free-runs at 6x drift (gyro bias ~0.2 deg/h class). Models were trained on v2b labels: dev only.
- Report `round5.txt`.

| Sat | solo local | solo fleet | fed fleet | pers fleet |
|---|---|---|---|---|
| SAT_1 | 0.384, 2/6, FA 3 | 0.663, 4/6, FA 2 | **0.768, 4/6, FA 1** | 0.333, 2/6, FA 4 |
| SAT_2 | 0.318, 24/30, FA 59 | 0.804, 30/30, FA 2 | **0.857, 27/30, FA 4** | 0.856, 23/30, FA 2 |
| SAT_3 | 0.521, 7/15, FA 6 | 0.815, 11/15, FA 2 | **0.849, 8/15, FA 0** | 0.814, 7/15, FA 0 |
| SAT_4 | 0.556, 1/5, FA 0 | 0.451, 3/5, FA 4 | **0.769, 2/5, FA 0** | 0.555, 1/5, FA 0 |

- Unseen-family recall solo local -> fed fleet: 0.20 -> 0.60, 0.75 -> 0.75, 0.44 -> 0.89, 0.20 -> 0.40.
- **Caveats:** the detector was designed while looking at this fleet's test half, so these numbers are optimistic. Most of the gain
  comes from fleet-selected settings; the weights' own effect is mixed at matched settings. SAT_1 pers is below solo fleet.

### RULE VIOLATION found in rounds 3-5 (found by audit, fixed before any fresh-seed scoring)

- `close` values of 240 and 720 rows and `gclose` of 240 and 2880 rows break the 70-row look-ahead limit (CLAUDE.md section 5): filling
  a gap between two alarms needs to know the second alarm is coming, up to 24 h ahead.
- **Results in `round3.txt`, `round3b.txt`, `round4.txt` and `round5.txt` that use those settings are invalid.** The round 4 and 5 tables above
  must not be quoted. They stay here because every round is logged, including bad ones.
- **Fix:** `close` limited to 10 and 60 rows; `gclose` replaced by a causal `latch` (0, 120, 240 rows) that only holds an alarm on after it fires.
- **Check added:** `fed/causal_check.py` recomputes the detector from data cut at row T and compares with the full run.
- Also fixed: `inputs()` cached `raw` paths could point at another fleet after a data move.

### Round 5b (rule-compliant rerun of round 5: causal latch, same data and models): the valid development result

- Report `fed/out_v2/reports/round5b.txt`. Past-only check `fed/causal_check.py fed federated`: PASSED, 12 cuts, 0 differences.
- Pre-declared headline, solo + local vs fed + fleet settings, anomalies only:

| Sat | solo local: F0.5, caught, false alarms | fed fleet: F0.5, caught, false alarms | unseen recall |
|---|---|---|---|
| SAT_1 | 0.384, 2/6, 3 | 0.664, 4/6, 2 | 0.20 -> 0.60 |
| SAT_2 | 0.318, 24/30, 59 | 0.853, 27/30, 4 | 0.75 -> 0.75 |
| SAT_3 | 0.521, 7/15, 6 | 0.594, 8/15, 5 | 0.44 -> 0.89 |
| SAT_4 | 0.556, 1/5, 0 | 0.586, 2/5, 1 | 0.20 -> 0.40 |

- Fleet totals: caught 34 -> 41 of 56, false alarms 68 -> 12. Fleet setting (fed arm): k2 fleet m30 s5 c10 h120 l3 W1m s15.
- **Caveats:** development fleet, detector designed while looking at it; models trained on v2b labels; SAT_3 fed fleet caught 0/6 of its own family
  and 8/9 unseen. Matched settings: pers >= solo on all four at solo's setting (0.908/0.714, 0.876/0.866, 0.851/0.593, tie), mixed at fed's setting.

### 60-day onboard window test (`FED_FIT_DAYS=60`, dev fleet v2c, causal detector): negative, not used

- Report `fed/out_w60/reports/w60.txt`. Anomaly F0.5 and caught, solo local / fed fleet / pers fleet:
  SAT_1 0.294 2/6, 0.523 4/6, 0.319 4/6; SAT_2 0.934 27/30, 0.846 29/30, 0.943 30/30;
  SAT_3 0.697 6/15, 0.630 8/15, 0.788 10/15; SAT_4 0.768 2/5, 0.398 2/5, 0.260 3/5.
- Solo became stronger, not weaker. Against solo local, fed fleet won 1 of 4 and pers fleet won 3 of 4 (SAT_4 fell to 0.260, below the 0.4 target).
  With the full window fed fleet won 4 of 4 and pers fleet 2 of 4. Mean F0.5: full-window fed fleet 0.674, 60-day pers fleet 0.578.
  Fleet totals caught 37 (solo) -> 43 (fed) -> 47 (pers) of 56, false alarms 6 -> 14 -> 23.
- **Decision (dev fleet only, before any fresh-seed scoring): the final run uses the full training window.**

### Design freeze for the final fleet (4 Nov 2026, before any fresh-seed scoring)

- **Final, full design frozen.** Full window, `fed/run_final.sh`, causal detector. Code sha256 (first 12) at freeze:

```
b64b13aacd63  fed/fedsim.py
edafe0230238  fed/rul.py
bfd30f849300  sim/config.py
33764095194c  sim/operations.py
e6d2e8565d50  sim/physics.py
172eae286bf8  sim/telemetry.py
630fba71c546  sim/products.py
63aa037620ce  ext/new_runs/core/period.py
429ea66b484e  ext/new_runs/core/detect.py
```


- **Dataset:** v2c code as of this entry, fresh seeds (`sim/fleet_final.json`), `SIM_FLEET` unset.
- **Detector:** `fed/fedsim.py` AXES, objective (mean of F0.5 all and anomalies), masks, ceilings; settings chosen only on validation.
- **RUL:** max-of-archive threshold, 30-day exponential fit.
- **Open:** whether training uses a 60-day onboard window (`FED_FIT_DAYS=60`) is decided on the dev fleet (`fed/out_w60`) before scoring.
- Pre-declared headline: solo + local selection vs fed + fleet selection, all arms reported. Scored once.

### Post-freeze bug fix (4 Nov 2026, before any fresh-seed detection score)

- The final run's detect stage crashed in the validation search: one candidate raised zero alarms on a validation window and ESA's metric
  code fails on an empty alarm list (`np.max` of an empty array). At the crash there were 0 test lines in the log and no results files.
- **Fix:** zero alarms scores F0.5 = 0 (`fedsim.f05`, `baselines.score`). This is what the metric would give; it changes no selection logic.
  RUL had already been scored on this fleet. Models and errors were kept; only detection was rerun.
- New checksums:

```
cbdd1936ceb3  fed/fedsim.py
7f078773d7a2  sim/baselines.py
```

### Final result (fresh seeds, frozen design, scored once; `results/final.txt`)

- Dataset: catch22 closeness 0.77, median coverage 0.99, momentum drift 2e-10. Anomalous rows 0.07-1.80 % (ESA 0.77-1.57 %); SAT_2 test 38.9 % with
  the run-to-failure event, 1.19 % without. Generation 53.7 min (shared with other jobs), round about 46 min + 4 min detection after the crash fix.
- Past-only check: PASSED for solo local, fed federated, pers federated (12 cuts each, 0 differences).
- **RUL:** alert day 256, correct wheel, failure day 356.1, lead 100.1 days. Exponential MAE 2.17, RMSE 2.44, alpha-lambda 0.83, PHM08 11.4, 15 % late.
  Min-rule 2.95 / 0.63. Straight line MAE 271. Prediction at 60/30/14/7 days before failure: error 0.0 / -4.1 / -2.7 / -2.0 days.

Anomaly-only unless stated. Fleet = federated selection.

| Sat | Arm | F0.5 all | F0.5 anom | P | R | Caught | Seen | Unseen | False alarms | FA time % |
|---|---|---|---|---|---|---|---|---|---|---|
| SAT_1 | solo local | 0.781 | 0.000 | 0.000 | 0.000 | 0/6 | 0.00 | 0.00 | 0 | 0.02 |
| SAT_1 | solo fleet | 0.909 | 0.833 | 0.999 | 0.500 | 3/6 | 1.00 | 0.40 | 0 | 0.07 |
| SAT_1 | fed local | 0.833 | 0.500 | 1.000 | 0.167 | 1/6 | 1.00 | 0.00 | 0 | 0.02 |
| SAT_1 | fed fleet | 0.795 | 0.555 | 0.666 | 0.333 | 2/6 | 1.00 | 0.20 | 1 | 0.06 |
| SAT_1 | pers local | 0.833 | 0.682 | 0.750 | 0.500 | 3/6 | 1.00 | 0.40 | 1 | 0.03 |
| SAT_1 | pers fleet | 0.892 | 0.832 | 0.832 | 0.833 | 5/6 | 1.00 | 0.80 | 1 | 0.13 |
| SAT_2 | solo local = fleet | 0.670 | 0.636 | 0.607 | 0.788 | 26/33 | 0.84 | 0.62 | 15 | 4.28 |
| SAT_2 | fed local = fleet | 0.878 | 0.870 | 0.854 | 0.939 | 31/33 | 0.92 | 1.00 | 4 | 3.54 |
| SAT_2 | pers local | 0.397 | 0.365 | 0.316 | 0.939 | 31/33 | 0.96 | 0.88 | 58 | 9.15 |
| SAT_2 | pers fleet | 0.923 | 0.919 | 0.921 | 0.909 | 30/33 | 0.92 | 0.88 | 2 | 1.73 |
| SAT_3 | solo local | 0.776 | 0.736 | 0.691 | 1.000 | 15/15 | 1.00 | 1.00 | 6 | 3.29 |
| SAT_3 | solo fleet | 0.774 | 0.686 | 0.777 | 0.467 | 7/15 | 0.33 | 0.56 | 2 | 0.08 |
| SAT_3 | fed local | 0.736 | 0.688 | 0.646 | 0.933 | 14/15 | 0.83 | 1.00 | 7 | 3.16 |
| SAT_3 | fed fleet | 0.915 | 0.882 | 1.000 | 0.600 | 9/15 | 0.33 | 0.78 | 0 | 0.03 |
| SAT_3 | pers local | 0.705 | 0.657 | 0.605 | 1.000 | 15/15 | 1.00 | 1.00 | 9 | 3.18 |
| SAT_3 | pers fleet | 0.845 | 0.784 | 0.888 | 0.533 | 8/15 | 0.33 | 0.67 | 1 | 0.05 |
| SAT_4 | solo local = fleet | 0.788 | 0.200 | 0.199 | 0.200 | 1/5 | - | 0.20 | 4 | 0.25 |
| SAT_4 | fed local | 0.307 | 0.073 | 0.060 | 0.600 | 3/5 | - | 0.60 | 47 | 0.69 |
| SAT_4 | fed fleet | 0.418 | 0.082 | 0.069 | 0.400 | 2/5 | - | 0.40 | 27 | 0.22 |
| SAT_4 | pers local | 0.823 | 0.238 | 0.250 | 0.200 | 1/5 | - | 0.20 | 3 | 0.19 |
| SAT_4 | pers fleet | 0.948 | 0.555 | 0.999 | 0.200 | 1/5 | - | 0.20 | 0 | 0.13 |

- Totals: solo local 42/59 caught, 25 false alarms; fed fleet 44/59, 32; pers fleet 44/59, 4.
- **Headline issue, must be disclosed.** The freeze entry named solo local vs fed fleet: F0.5 improved on 3 of 4; SAT_4 fell 0.200 -> 0.082 with 27 false alarms.
  The original design (§3) named solo local vs pers fleet: improved on 4 of 4, all above 0.4 (0.832, 0.919, 0.784, 0.555). Choosing pers after seeing these
  numbers is post hoc, even though §3 predates this session. Both are scored results; report both.
- **Losses:** SAT_3 pers fleet caught 8/15 against solo's 15/15, seen-family recall 1.00 -> 0.33. SAT_4 pers fleet caught only 1/5; its 0.555 is precision.
  Out-of-limit baseline, anomalies only: 0.500, 0.372, 0.370, 0.588, so SAT_4's 0.555 does not beat limit checks.
- Weights at equal settings, pers vs solo: better in 9 of 12 comparisons (4/4 at pers's setting, 3/4 at solo's, 2/4 at fed's). Stronger than on the dev fleet.
- Single seed, few events per satellite: one event moves a score a lot.

## 6. How to run (any machine)

### 6.1 What is in git and what is not

- **In git (repo = this `New/` folder, ~2 MB):**
  - `sim/`: the generator;
  - `fed/`: experiments;
  - `ext/new_runs/{core,vendor}`: a **frozen copy** of `ESA/Main/new_runs` code from 4 Nov 2026 (the
    GDN2 model, the detection stack, ESA's metric package). Edit the original in `ESA/Main` for ESA work;
    the code here prefers `ext/`. Override with `NEW_RUNS=/path/to/new_runs`.
  - `envs/*.yml`: exact conda environments (exported on macOS);
  - `sim/fleet_v2b.json`: the frozen seeds of dataset v2b;
  - `CLAUDE.md`, `REPORT.md`, `HANDOFF.md`.
- **Not in git** (`.gitignore`):
  - `data/`: the dataset (1.2 GB), regenerated from the seeds;
  - `fed/out_*`: models, errors, results;
  - `*.npy`, `*.pt`, `*.pkl`, Vizard `.bin` files.
- **ESA reference statistics are in git, so no ESA data is needed on a new machine:**
  - `sim/esa_ref_catch22.npy`: catch22 features of the same 253 real 1-day windows (M1+M2 lightweight,
    seed 0). `validate.py` uses it when the ESA arrays are absent.
  - `sim/esa_ref_labels.json`: ESA label statistics for the lightweight channels.
    - **M1:** 0.77 % anomalous time, 3.7 anomalies/yr, 4.6 rare/yr, 94 % multivariate-global.
    - **M2:** 1.57 %, 5.4/yr, 173 rare/yr, 47 % multivariate-global / 47 % multivariate-local.
- **ESA-ADB data is not needed** for the fleet work.
  - The only use is the catch22 similarity check in `validate.py`, which is skipped if
    `ext/new_runs/data/M1_train_vals.npy` is missing.
  - To include it, copy `ESA/Main/new_runs/data/M{1,2}_train_vals.npy` (310 MB) into
    `ext/new_runs/data/`.

### 6.2 New machine setup (once)

```
git clone <your private repo> Capstone/New && cd Capstone/New
# conda / miniforge installed
conda create -n bsk python=3.11 -y && conda run -n bsk pip install bsk==2.12.2 pycatch22==1.0.2 pandas matplotlib scipy
conda create -n timeeval python=3.9 numpy=1.21.6 pandas=1.5.3 scipy=1.11.4 scikit-learn=1.6.1 matplotlib=3.5.3 statsmodels=0.12.2 -y
conda run -n timeeval pip install torch==2.8.0 numpyencoder==0.3.2     # Linux + CUDA: use the CUDA wheel from pytorch.org
conda run -n bsk python -c "import Basilisk; print('ok')"
```

- If an install fails, `envs/*.yml` lists every package version. Those files were exported on macOS, so
  use them as a reference rather than with `conda env create` on Linux.
- Device choice is automatic (`core/train.pick_device`: CUDA, then MPS, then CPU).

### 6.3 One round

```
mkdir -p data/sim_fleet && cp sim/fleet_v2b.json data/sim_fleet/fleet.json    # same dataset as round 2
cd sim && caffeinate -i nice -n 10 conda run -n bsk python run_fleet.py --workers 7 --prod-workers 2 > ../data/gen_v2.log 2>&1
cd .. && caffeinate -i zsh fed/run_round.sh > fed/round.out 2>&1      # validate, baselines, RUL, solo/fed/pers + detection
```

- **On Linux:** drop `caffeinate -i`, and use `bash` with `fed/run_round.sh` (the script works in bash).
- **More cores:** `--workers 14` runs all physics jobs at once (each needs ~1 GB RAM).
- **This machine / any borrowed machine:** `source .localenv/activate.sh`, then call the env Python directly. `fed/run_final.sh` runs the whole final pipeline.
  Generation `--workers 14 --prod-workers 4` is 24.5 min alone on an M5 Max; one round is about 46 min; detection alone about 4 min.
- **Fresh dataset (final frozen run):** do **not** copy `fleet_v2b.json`. `run_fleet.py` then draws new
  seeds and writes `data/sim_fleet/fleet.json`; keep a copy in `sim/`.
- **Outputs:**
  - `fed/out_v2/results.json`: all arms and both selection modes, per satellite;
  - `fed/out_v2/log.txt`;
  - `fed/out_v2/rul.json`, `rul_SAT_2.png`;
  - `data/sim_fleet/validation/*.json`.
- **Restarts:**
  - `run_fleet.py` reuses physics if `data/sim_work/<sat>/phys/physlog.json` exists. Delete `data/sim_work`
    after changing physics or timeline code.
  - `fedsim.py` skips finished stages (cache, models, errors, results). Delete `fed/out_v2` for a clean
    round, or delete only `results.json` to redo detection alone (~5 min).
- **Timing measured on the old 16 GB Mac:** generation 70 min; training 110–140 s per epoch, so solo 4 ×
  27 min, then fed and pers; the full round ≈ 4 h.

## 7. Levers if results are poor (agreed with the user; cheapest first; log every change)

1. **Detector only, no retraining:** extend the search (Telemanom pruning as on M1, more thresholds).
   Delete `results.json` and rerun the `detect` stage.
2. **Easier but realistic faults:**
   - larger gyro bias drift (`_params` B1 amp) or longer solar-string opens (C1);
   - fewer calibrations on SAT_1–3 (`RARE_RATE`).
   - Sensor and power faults are post-processing, but this still needs regeneration with new seeds.
3. **No federated learning gain:** a shorter local training window per satellite (onboard storage, e.g.
   the last 60 days before validation). This is the classic case for FL. Implement it as `fit_end`
   start-offset rows in `fedsim.py`, the same for every arm.
4. **Final:** freeze the design, regenerate with fresh seeds, score once, and report all arms.

## 8. Known caveats

- Single seed.
- Selection relies on a few validation events per satellite (3–5).
- `sim/validate.py` and `sim/baselines.py` are updated for v2.
- `sim/report.py` (README/DATASHEET/VALIDATION writer) and `sim/viz_clips.py` (Vizard clips) still
  describe v1 (SAT_5, 34 channels). Update them before the write-up.
- Physics on a different CPU architecture (x86 vs Apple ARM) may differ at floating-point level, so
  dataset v2b is near-identical, not bit-identical. Fine for development; the final frozen run is
  regenerated anyway.

## 9. Conversation context: decisions, reasoning and the user's preferences

A new chat has none of the earlier conversation, so this section holds it.

### 9.1 How the user works

- **Rounds:** go stage by stage. After each round, report a compact table per satellite:
  - solo vs fed vs pers;
  - F0.5 (all events and anomalies only), precision, recall;
  - recall on fault families seen vs unseen locally;
  - RUL lead time and error.
  Then **ask** whether it is good enough. If the results are clearly bad, diagnose and fix first, then
  report.
- **Long jobs:** tell the user before starting installs, long runs or anything beyond the agreed stage.
- **Style:** short, plain, honest answers; tables welcome; no hype. Be token-efficient: the weekly limit
  is tight, so avoid reading large files whole and keep tool calls few.
- **Machine:** keep it responsive (nice, few threads, memory watch). On macOS use `caffeinate -i`: a
  sleep once stalled a run for ~5 h.
- **Ease off if needed.** If results are hard to get, make the dataset slightly easier rather than adding
  complexity. Keep the ESA-ADB format, labels and event types, mostly-local anomalies, ESA metrics,
  baselines and the validation framework.
- **Logical consistency matters.** The user doesn't want to explain every step; check the logic yourself
  (no leakage, no test tuning, a causal story that holds).
- **No cloud money.** Free compute is: the old Mac (16 GB), an **M5 Max (64 GB)**, the **RTX 4500 Ada PC**
  (Ubuntu, 32 threads, 128 GB, 24 GB GPU) and Kaggle (one account).
  - Estimated full round: old Mac ~5.5 h; M5 Max ~1.5–2 h; RTX ~45–75 min.
  - Estimated worst case of 5 rounds: ~6–8 h on the new machines.

### 9.2 Targets agreed with the user

- **Detection:** a decent solo per-satellite F0.5 (for example ~0.5–0.6) that improves with FL (for
  example to ~0.6–0.7). **Even a marginal gain is fine.** The result should be realistic and promising,
  not rigged.
- **RUL** must work. The "decent → better" story for RUL:
  - **alone,** a satellite can only raise an alert (z > 6), because it has never seen a wheel fail;
  - **with fleet knowledge** (the failure threshold from the 3 archived run-to-failure wheels of the same
    design), it predicts the failure date. Report both.
- **The story the user wants for FL:** a satellite trains on one fault family. In test it meets two and
  misses the one it never saw. After FL it detects both, so its score improves. **Logic used to make this
  real:**
  - GDN2 is unsupervised (it forecasts normal behaviour; labelled rows are masked from the loss). So
    "having seen a fault" matters through the **detector settings** chosen on the satellite's validation
    window, and through how well the model forecasts normal behaviour.
  - FL therefore contributes in two ways:
    1. **Shared weights** trained on 4 satellites' normal data (a lower error floor).
    2. **Federated detector selection:** each satellite scores the candidate settings on its own
       validation events and shares only one F0.5 per setting. The fleet picks the setting with the best
       mean, and each satellite applies it to its own robust-z error scale. This is federated
       hyperparameter tuning: no telemetry or labels leave the satellite.
  - The dataset ensures every satellite misses at least one family in training, and every family is seen
    in training somewhere (§2 table).
- **Round 1 caveat:** solo already caught 60–80 % of unseen families, because large deviations are
  caught anyway. If round 2 shows the same, the honest options are:
  - make some faults subtler (realistic: operators miss exactly those);
  - use a shorter local training window (§7.3);
  - or report the FL gain as precision (fewer false alarms).
  Do not tweak the test data until it "works": decide on validation or a development fleet, then run the
  final frozen fleet once.

### 9.3 Decisions and why

- **No fault-type classifier for now** (the user chose this; it would add complexity).
  - The fault family is attributed from the alarmed channels (`fed/fedsim.py` `fam_of`).
  - Labels are complete, so a classifier can be added later without regenerating.
- **Health states vs health score:**
  - The **states** (healthy / degrading / failed, plus phases and RUL) are ground truth in
    `truth/health.csv`.
  - The **score** is the model's output from telemetry. Comparing the two proves the score works.
  - The user said to keep it simple: the RUL health indicator is the wheel excess-current fraction (§4).
  - A fleet-wide ground "health score" formula was proposed but **not built**:
    H = 1 − maxₛ(wₛ·Rₛ), where Rₛ = recency-decayed (τ ≈ 7 d) severity × duration of alarms + a
    degradation trend.
- **Event rate:**
  - ESA's raw rate (3.7–5.4 anomalies per year) would give too few events in a 1-year satellite, so we use
    roughly 10–27 anomalies per satellite-year ("ESA compressed").
  - We match ESA in anomalous-time share (~1–2 %, except SAT_2 test because of the long degradation),
    random arrivals, varied durations and mostly local anomalies.
  - Reference: `sim/esa_ref_labels.json`.
- **The user's per-anomaly "optimal detector stack" idea:**
  - Legitimate only on training and validation data, never on test.
  - Combining per-event optimal settings with OR would add up false alarms. The correct form is to use
    per-event diagnosis to narrow the search, then pick **one** setting that maximises F0.5 over all
    validation events (as `search()` does).
- **FL personalisation:** shared layers are averaged; channel and command embeddings, normalisation and
  detector thresholds stay local; `pers` = a short local fine-tune after FL. This follows the CLAUDE.md
  rule never to average whole models.
- **Communication gaps** are excluded from the event types (the user's request). All telecommands are
  priority 3. No operator recovery modelling. FDIR attributes are in `events_truth.csv` (component,
  subsystem, severity tier 1–3).
- **Compute:** the old Mac (MPS) trains at ~110–140 s per epoch of 150 batches. One GPU job at a time;
  4 parallel processes per GPU are possible on bigger machines (needs a small change to `fedsim.py`).

### 9.4 Open items (not done)

1. Round 2 results, then report to the user and ask.
2. If needed: levers in §7, then the **final frozen fleet** (fresh seeds, scored once).
3. Update `sim/report.py` (README / DATASHEET / VALIDATION for v2) and `sim/viz_clips.py` (Vizard clips:
   normal operations with profiled slews, SAT_2 wheel failure, the fleet). The v1 clips used SAT_5.
4. Write-up of the FL and RUL results, kept separate from the ESA-ADB results (CLAUDE.md rule).
5. Later stages from CLAUDE.md §8: federated operator feedback, closed-loop FDIR, int8 on Cortex-A53, a
   dashboard view.
6. **The v1 dataset** (5 satellites, 34 channels; too easy for limit checks) exists only on the old Mac
   (`data/sim_fleet_v1/`). It is not needed.
