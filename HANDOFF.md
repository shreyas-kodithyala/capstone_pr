# HANDOFF: Basilisk fleet v2, federated learning and RUL (living log)

Read first: `CLAUDE.md` (rules and context), then this file. It records **what was done, why, where
it is, and what is next**, so work can resume from here. The newest status is at the top of §0.

## 0. Status (newest first)

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
