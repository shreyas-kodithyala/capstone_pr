# CLAUDE.md: Capstone 122, onboard federated satellite health monitoring

Read this first, then **`HANDOFF.md` (current work, newest status, how to resume)**, then `REPORT.md` in this folder: the full walkthrough of the model, every result,
and the detailed plan. This folder (`Capstone/new`) is for the **next phase**: the Basilisk
simulated fleet, federated learning, onboard int8, and FDIR. The validated detector and all
earlier work live in `/Users/Shreyas/Downloads/Capstone/ESA/Main/` (see that folder's own
`CLAUDE.md` for data facts and environment landmines).

## 1. The project in one paragraph

PES University capstone ("Federated Learning Architecture for Health Monitoring and Anomaly
Detection in Multi-Satellite Systems"; team of 4, guide Dr. Richa Sharma; first review 3 Nov 2026).

- **The detector:** a graph-temporal forecaster (GDN2: TCN + learned directed dynamic graph +
  GATv2 + telecommand cross-attention, 202k parameters), plus a past-only detection stack.
- **Validated on real ESA telemetry (ESA-ADB)**, where it beats every published result:
  - M1 lightweight **0.866** vs 0.786 published;
  - M2 lightweight **0.987** vs 0.949 published (SMED 0.971).
- **Next phase:** use the Basilisk spacecraft simulator to demonstrate what real single-satellite
  data cannot:
  - a fleet of same-design satellites;
  - federated learning between them (rotating leader, robust aggregation, personalisation);
  - operator feedback shared across the fleet;
  - degradation early warning;
  - closed-loop tiered FDIR;
  - onboard int8 feasibility (Cortex-A53).

**Framing:** ESA-ADB = real-world validation; Basilisk = system demonstration. Never mix their
numbers.

## 2. Problem statement (agreed, scope-checked)

> Satellite health monitoring today relies on static onboard limit checks and on ground-based ML
> over downlinked telemetry, which is slow, bandwidth-limited, prone to false alarms, and cannot
> learn from faults seen elsewhere in a fleet. This project develops an onboard, federated
> health-monitoring architecture for multi-satellite systems. Each satellite runs a lightweight
> (int8, ~0.2 MB) graph-temporal anomaly detector that models dependencies between sensors and
> telecommands, flags the affected component, and triggers a tiered FDIR response. Satellites
> improve their models collaboratively by exchanging model updates (never raw telemetry) over
> inter-satellite links, aggregated by a rotating leader satellite with robust filtering of faulty
> nodes; ground operators review alarms and their verdicts feed back into training. Detection is
> validated on real telemetry (ESA-ADB), where the detector exceeds published baselines; fleet
> learning, degradation early warning and closed-loop FDIR are evaluated in a Basilisk-based
> multi-satellite simulation; onboard feasibility is measured by running the quantized detector on
> an ARM Cortex-A53. cFS integration and LEON-class processors are future work.

**Out of scope for now:**
- porting to NASA cFS or LEON3 (weeks of C work);
- claiming onboard training on flight hardware without measuring it;
- claiming "first" without a proper literature check.

## 3. Results to quote (details and caveats in REPORT.md §6)

| Result | Number |
|---|---|
| M1 lightweight, ESA event-wise F0.5 (ch 41–46, test 2007–2014) | **0.866** (P 0.974, R 0.600); alarm precision 0.886; ADTQC 0.255 (weak) |
| M2 lightweight (ch 18–28) | **0.987** (P 0.985, R 0.997) |
| Operator in the loop, M1, monthly retraining with simulated operator feedback, 2007–2013 | **0.888 vs 0.866 frozen**; better on all ESA aspects. Ablation (same months): frozen 0.860, no feedback 0.876, with feedback 0.883. Setting (lr 3e-4) chosen on a 2005–2006 rehearsal |
| Transfer to a "new satellite", M1, last N days only | 14 days: scratch 0.885, from M2 **0.891**, M2 with TCN frozen 0.872, from M1_S3 0.879. 63 days: scratch 0.870, M2 0.870, frozen 0.853, M1_S3 0.854 |
| Per-subsystem | M1_S3 0.870 · M1_S6 0.679 · M1_S1 0.600 · M2_S2 0.025. M1_S5 result not collected; M2_S1/S5/S6 not run |
| Full channel set M1 (cloud) | 0.023: false-alarm time from OR-ing 58 channels. A reported negative result |
| Old federated prototype (older model) | M2 0.923 → 0.936, M1 0.080 → 0.018. Not yet run on the current model |

All results are single seed unless stated.

## 4. Where the code is

- Pipeline: `ESA/Main/new_runs/`:
  - `core/model.py` (GDN2; `shared_state()` excludes `embedding.*` and `cmd_emb.*`, the
    federated split);
  - `core/train.py`, `core/features.py`, `core/detect.py`, `core/period.py`, `core/select.py`,
    `core/evaluate_par.py`, `core/esa.py` (ESA's metric classes, vendored);
  - `core/online.py` + `run_online.py` (operator loop);
  - `run_tl.py` (transfer learning);
  - `run_sub.py` (subsystems);
  - `export_dashboard.py` + `dashboard/` (operator console on localhost:8765).
- Config switches (environment variables):
  - `NR_SET` = light | full | sub;
  - `NR_SUB`;
  - `NR_TRAIN_DAYS`, `NR_TRAIN_FROM_END=1`;
  - `NR_INIT` (load shared weights);
  - `NR_FREEZE=tcn`;
  - `NR_OUT`, `NR_DATA`.
- Trained lightweight models: `ESA/Main/out 2/{M1,M2}/final_s0/model.pt` (+ meta.json with mu/sd);
  detector settings in `out 2/{M1,M2}/frozen.json`.
- Basilisk sample: `ESA/Main/sim/sample_rw_scenario.py` (4 HR16 wheels, slew command; Vizard file
  in `sim/_VizFiles/`).
- Environments:
  - `conda activate timeeval` (Python 3.9, numpy 1.21.6, torch) for the ESA pipeline;
  - `conda activate bsk` (Python 3.11, Basilisk 2.12.2, pytest, matplotlib) for the simulator.
- Vizard 2.4 is installed on the Mac (open a `.bin` from `_VizFiles`).

## 5. Rules (never break)

- Never feed `is_anomaly` / labels as model input; labels only mask the loss.
- No thresholds or settings tuned on the test split; choose on validation folds or a rehearsal
  period, then score the test once. No oracle numbers next to published baselines.
- Detection must be past-only (look-ahead at most 70 rows); verify by recomputing alarms in one
  pass.
- Keep Telemanom pruning on M1 and off M2; no single global score broadcast to all channels.
- Federated learning averages **shared layers only**; embeddings and detector settings stay
  per-satellite. Never naively average whole models across different spacecraft.
- No Mission 3. No PyPI `timeeval`. Don't cite DFSat (retracted); don't quote STGLR's 0.97.
- Don't commit datasets. Don't edit `ESA/Main/src/run_m1_lightweight.py` (provenance of an older
  result).
- Simulator results are reported separately from ESA results. Fix the scenario, faults, seeds and
  metrics **before** training; report every arm, including ones that lose.
- Report all ESA aspects, not just the headline. Say plainly when something didn't work.

## 6. How the user works (important)

- **Don't start runs, installs or long jobs until the user says go.** Propose, then wait. They have
  said so firmly.
- **Keep the Mac responsive:** it has 16 GB RAM. Use nice, few threads, one GPU job at a time, and
  a memory watch. They were upset when runs ate all the memory.
- **No money left for cloud.** Free compute only: the Mac, Kaggle (one verified account), and the
  RTX PC.
- Plain, concise, honest answers; no hype. Tables are welcome. Give periodic updates during long
  runs, and never be a black box.
- Highest quality matters, but within the machine limits and the deadline.
- The Mac's Apple GPU (MPS): use the fixed `to()` (non_blocking only on CUDA). Older MPS error files
  were corrupted.

## 7. Machines

- **MacBook:** M-series, 16 GB, about 35 GB disk free. ESA data and everything above live here.
- **RTX PC:** Ubuntu 24.04, i9-14900 (32 threads), 128 GB RAM, 1 TB, RTX 4500 Ada (24 GB).
  - Best for the fleet simulation and training.
  - Access via SSH (VS Code Remote-SSH; Tailscale if off-network). AnyDesk only for viewing
    Vizard.
  - Nothing is installed there yet.
  - For the simulator alone no files need copying. To run the detector there, copy the code bundle
    (`ESA/Main/new_runs_sub_code.tar.gz`, regenerate it with `make_sub_kaggle.py` first, since it
    predates today's fixes and online/TL code). Copy the data archive only for ESA work.

## 8. Next steps (agreed order; confirm with the user before each)

1. **Simulator realism (Basilisk):**
   - 3 same-design satellites with small variations;
   - about 15–20 channels at 30 s with noise and dropouts;
   - scripted telecommands;
   - fault and degradation library with exact labels (wheel friction rise, wheel stuck, gyro bias
     drift, stuck sensor, dropout, battery fade);
   - each satellite trains on a different normal mode and a different fault; the test has all;
   - export in ESA-ADB format;
   - physics sanity checks.
2. **Run the existing detector on the fleet:** solo vs federated vs federated + personalisation vs
   centralised. Metrics fixed in advance (REPORT.md §8.3), including recall on faults first seen
   elsewhere.
3. Federated operator feedback; degradation early warning (lead time); transfer to a replacement
   satellite.
4. Closed-loop tiered FDIR in Basilisk; int8 quantization (fp32 vs fp16 vs int8 on the ESA test)
   and Cortex-A53 timing; dashboard view; write-up.

**Also pending:**
- M2_S1/S5/S6 subsystem runs (RTX PC);
- 3 seeds for the headline results.
