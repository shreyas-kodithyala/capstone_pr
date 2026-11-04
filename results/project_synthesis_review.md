# Whole-project review: story, novelty, competing scores, use cases

Date: 4 Nov 2026. Covers all six work items. Companion to `paper_readiness_review.md`.

ESA/Main is not in this folder, so the ESA-side numbers below come from `REPORT.md` and the team's pasted tables. They were checked for internal consistency and against the published ESA tables, but not rerun.

## 1. Verdict

- Strong capstone with honest evaluation. Not yet a top-tier paper.
- "Beats every published result" is no longer true on Mission 1. Goetze et al. report 0.927 at IEEE SCC 2025.
- Mission 2's 0.987 is a real win, but on a subset ESA itself calls relatively trivial.
- The best contributions are the hidden-anomaly (zero-order-hold) finding, the calibration evidence and the fleet dataset, not the GDN2 architecture or FedAvg.

## 2. Papers with higher or nearby scores, and how they scored

| Work | Score | Data and split | Scoring | Comparable? |
|---|---|---|---|---|
| Kotowski et al. (ESA) | M1 0.786, M2 0.949 | Official split, lightweight, no telecommands | Official code. Deliberately not tuned | Same metric and test, but we used extra inputs and more tuning |
| **Goetze, Schlippe, Lakey (IEEE SCC 2025)** | **M1 0.927** (P .998, R .723); 29k-parameter version 0.888 | M1 lightweight. Validation about 265,000 rows, which equals ESA's 3 months at 30 s. Five of six recalls are whole counts out of 65 events (47/65 for the best) | Says it uses ESA's CEF0.5. Official code not stated. Describes a 60 s merge tolerance. Event-wise only. No seed count. An ESA-ADB co-author is on this paper | **Very likely the same test. Treat it as beating 0.866 unless reproduced** |
| Allegrini & Pompei (ML4ITS 2025) | Public 0.931, private 0.853 (3rd). Post-hoc best private 0.90 | Kaggle hidden test, described as 58 of 76 channels. Supervised, shapelets mined from labelled anomalies | Corrected event-wise F0.5 | Not comparable: different test and task. Our M1 full-set result was 0.023 |
| Structure-aware EVT (arXiv 2609.10017) | M1 0.700, M2 0.698 | Whole mission, month-by-month retraining, no labels | Corrected event-wise F0.5, 6 h alarm grouping | Different split, lower score |
| Benchmark study (arXiv 2607.07335) | M1 0.448, M2 0.712 | Own split for unsupervised models | Plain event overlap, own metric | Not comparable, lower score |
| Rudd (GitHub, unreviewed) | Tuned Telemanom 0.798 on ch 41-46 | Own 30 s grid | "Mission-level" F0.5 | Not comparable, but shows tuning alone moves far from ESA's 0.178 |
| Sempere (GitHub, unreviewed) | M2 about 0.97-0.98 with simple models | Own online protocol, label-tuned at the top level | Own implementation, global alarms | Not comparable, supports M2 being easy |
| "SMED 2025" | 0.801, 0.971 | Not found | Not found | **Three searches found no source. Find the citation or remove it** |

The sota2 leaderboard mixes Allegrini's public and private columns, so its ranking should not be cited.

Rough 95% ranges on M1 from event counts alone (precision held fixed): ours 0.81-0.91, Goetze 0.88-0.96, ESA 0.70-0.86. Ours and Goetze overlap, so 8 extra events is suggestive rather than proven. Seeds would add further spread.

### Where our protocol differs from ESA's lightweight protocol

1. **Telecommands.** ESA: "The lightweight subsets do not include any telecommands." GDN2 used 30 (M1) and 123 (M2), per `REPORT.md` 4.1.
2. **Availability features.** `count`, gap flag and `tsince` come from raw timestamps that baselines did not get. This is defensible as ESA's requirement R8 but it is not like-for-like.
3. **Tuning.** ESA's baselines use default settings on purpose. Ours used a labelled coordinate search on two folds. ESA's R4 permits learning from labels in training data, so this is legal, but it is more tuning.
4. **Test reuse.** `REPORT.md` shows scoring across phases (0.853 to 0.866, 0.964 to 0.987) and says the pipeline's shape came from logged iterations. Count how many times test was scored and disclose it.
5. **Silence finding.** Confirm whether the 56.7% statistic came from training years only.

### Corrections to the pasted claims

- "Match the benchmarks on other metrics": true against ESA's top-ranked algorithm, not against the best per metric. On M1, channel-aware F0.5 is 0.592 vs 0.837, and ESA treats channel identification as a primary aspect. ADTQC is 0.255 vs 0.901.
- "Built from scratch": GDN2 is new, but the stack also reproduces ESA's Telemanom error rules, including its 0.007 minimum error.
- Subsystems "as it is, not fine-tuned": `REPORT.md` says one model per subsystem with up to 8 context channels, and M1_S6 had its own detector search. The detector settings were reused for the rest, but models were retrained.
- M1 anomalies only: ESA 0.968, ours 0.939 frozen and 0.968 retrained. Both catch 25 of 29.
- M2 test anomalies: `REPORT.md` says 1, ESA's table says 4 (recalls in its anomalies-only table are only 0 or 1). Check this before quoting 0.381.
- All ESA figures in the pasted table were checked against Table 2 and match.

## 3. New findings on the fleet result

These come from `fed/out_final/results.json` and logs, with no new training.

### Where the gain comes from

| Step | Mean anomaly F0.5 | Caught of 59 | False alarms | Change in F0.5 |
|---|---|---|---|---|
| Solo, own settings | .393 | 42 | 25 | |
| Solo, fleet settings (shares one score per setting) | .589 | 37 | 21 | +.196 |
| Plain FL weights, fleet settings | .597 | 44 | 32 | +.008 |
| FL + fine-tune, fleet settings | .773 | 44 | 4 | +.176 |

Each arm selected its own fleet setting, so the steps are not perfectly controlled.

- Sharing settings and local fine-tuning produce the score gain. Averaging weights adds almost nothing to score, though it adds 7 catches and 11 false alarms.
- **Plain FL forecasts worse than solo on all four satellites** (validation loss .0678/.0730/.0381/.0534 vs .0533/.0582/.0329/.0348). It was still improving at round 12, the cap.
- Fine-tuned FL beats solo forecasting on 2 of 4 satellites only.
- Fine-tuning adds 100 batches at lr 3e-4 on top of up to 1,800. Solo has no matched extra training, so this control is missing.

### Faults a satellite never saw in training

This is the metric `REPORT.md` 8.3 fixed in advance. Each family was seen somewhere in the fleet.

| Arm | Unseen locally | Seen locally |
|---|---|---|
| Solo, own settings | 15/27 | 27/32 |
| Solo, fleet settings | 13/27 | 24/32 |
| Plain FL, fleet settings | 18/27 | 26/32 |
| FL + fine-tune, fleet settings | 18/27 | 26/32 |
| Plain FL, own settings | 20/27 | 29/32 |
| FL + fine-tune, own settings | 19/27 | 31/32 |

FL weights add about 5 catches on unseen faults (13 to 18 at fleet settings, 15 to 20 at own settings). This is FL's cleanest benefit, but it is 5 events on one fleet. SAT_4 stays poor (1-3 of 5).

### Rule conflict

`CLAUDE.md` section 5 says detector settings stay per-satellite. The headline arm uses fleet-selected settings. I did not flag this earlier. Under the written rule the compliant comparison is .393 to .486 (caught 42 to 50, false alarms 25 to 71). Either change the rule openly and call fleet selection a contribution, or report the compliant arm as primary.

## 4. One story

Thesis to test: **in spacecraft monitoring, calibration and deployment decide results more than the forecaster does, and calibration is what a fleet should share.**

Supporting evidence:

- Same encoder, opposite rules: pruning gives M1 0.853 and M2 0.014; absolute z gives M2 0.964 and M1 0.137.
- 14 days of training (0.885) matches 7 years (0.866).
- M1 full set collapsed to 0.023 through false-alarm time.
- On the fleet: no command mask catches 9 of 59; one absolute threshold gives 919 false alarms; settings chosen without a satellite's own data score 0.776.

Counter-evidence: Goetze's better forecaster under a Telemanom-style threshold reaches 0.927. So the thesis needs a forecaster-swap ablation.

Lifecycle framing for the report:

| Stage | Question | Work item | Evidence type |
|---|---|---|---|
| Launch | Can it start with little data? | Transfer | Real, emulated new satellite |
| Operations | Does it detect with few false alarms? | GDN2 + stack | Real |
| Ageing | Can it stay current safely? | Operator retraining | Real, simulated operator |
| Scale | Does it work beyond 6 channels? | Subsystems | Real, 4 of 8 done |
| Fleet | Does sharing help? | Federated | Simulated |
| End of life | Can it warn before failure? | RUL | Simulated |

Missing bridges:

1. The ESA and fleet detector stacks differ. Run the noise-ceiling and command-mask calibration on ESA data.
2. FL has never run on the current model with real data. Try S5 and S6 pairs across M1 and M2, accepting that these are different spacecraft.
3. No detector-to-RUL handoff, replacement satellite test or federated operator feedback exists yet.

Claims to drop until done: robust filtering, int8 and Cortex-A53, closed-loop FDIR, inter-satellite links, "flags the affected component" on the fleet (attribution 0-.77).

## 5. Novelty by item

| Item | Closest prior work | Novelty |
|---|---|---|
| Hidden anomalies in resampled telemetry (about 30% of M1 anomalies) plus silence detector | ESA states the requirement (R8) and no ESA algorithm meets it. Found no paper quantifying it | **Highest on real data**, needs an ablation |
| Fleet dataset: physics simulation, ESA format, exact labels, run-to-failure | Carneiro et al. 2022 (Basilisk wheel faults, single spacecraft), OPS-SAT-AD (real, single), DLR synthetic telemetry, STA-FedAvg synthetic proof of concept | **Highest overall**, say "not aware of" |
| Calibration-first finding across real and simulated data | Laridi et al. for federated threshold selection | Medium, as a finding rather than a method |
| Archive-based fleet RUL threshold | Federated RUL exists for aircraft engines. Found none for reaction wheels | Medium as an idea, weak evidence |
| GDN2 | GDN, TCN, GATv2. ESA already feeds telecommands to full-set models | Low-medium: telecommand cross-attention |
| Operator retraining | JPL feedback-loop work, EVT monthly retraining | Low-medium: useful safety gate, small effect |
| Transfer | CNES/TeSA 2023, Baireddy 2021 | Low |
| Subsystem models | Standard design | Low |
| FedAvg + fine-tune | Atefrad & Karami 2025 | Low alone. GNN + FL + personalisation for satellites was not found, but that is only a combination |

## 6. Problems, use cases and impact

| Problem | What we show | Strength |
|---|---|---|
| False alarms waste operator time | M1 alarm time about 1%, precision .974. S6: 25 events at 0.18% alarm time; fixed limits catch 28 at 25% | Real data, single seed |
| Resampling hides anomalies | Availability features plus silence detector | Real data, needs ablation |
| Commands look like faults | Masks and telecommand input. Removing masks collapses detection in the sim | Simulated, plausible on real data |
| New satellites have little data | 14 days reaches about the same score | Real, emulated, single seed |
| Models go stale | Past-only retraining with a gate. 77 of 83 updates kept, no year clearly worse | Real, small effect, simulated operator |
| A satellite has not seen a fault | About 5 more of 27 unseen faults caught with FL weights | Simulated, one fleet |
| Settings are hard with few local faults | Leave-one-satellite-out 0.776 | Simulated, one fleet |
| Wheel failure surprises | 100-day lead in a best-case setup | Simulated, circular |
| No public fleet fault data | Dataset | Exists now, needs release |

Impact, judged honestly:

- Today: a research prototype with no operational use.
- Near term: the dataset, if released with a datasheet, and the hidden-anomaly finding could be used by other researchers.
- Medium term: ground-side alarm triage is the most practical deployment. FL applies mainly when data cannot be pooled, for example several owners or onboard autonomy without downlink. A single-operator fleet can centralise, and FL moves about 19 MB vs about 32 MB raw here.
- Onboard use is unproven until int8 and timing are measured.

## 7. How real the simulated parts are

**Basilisk dataset**
- Real: orbit, attitude, wheels, eclipse, power and battery physics pass checks; ESA format; exact labels; realistic share of anomalous time.
- Not real: 58% multivariate-local vs 94% multivariate-global on ESA M1; perfect navigation, so sensor faults do not reach control; 73% catchable by a single hindsight threshold; faults scripted and tuned in FL's favour during development (allowed, must be disclosed).
- Rating: physically credible, statistically a different mission.

**Federated learning**
- Real: actual FedAvg of shared layers, separate satellite processes, no telemetry exchanged, fresh seeds, scored once, past-only checks passed.
- Not real: all on one laptop; no links, contact windows, latency or dropouts; leader rotates in the log while averaging runs in one process; no faulty nodes; one fleet, one seed.
- Rating: real algorithm, simulated network.

**RUL**
- Real: causal pipeline from alert to failure estimate, with the threshold learned from archive satellites.
- Not real: one failure event; same wheel, same exponential shape, same failure level as the archive, and the predictor also fits an exponential.
- Rating: a working demo. The 2.2-day error is close to a best case and should not be quoted as accuracy.

## 8. Next work, in order

1. Reproduce Goetze on the same test with the official scorer.
2. ESA reruns: no telecommands, then no availability features, then forecaster swap. Three seeds. Needs ESA data.
3. Fix the RUL design, then 5 fleets x 3 seeds, with centralised and matched-budget solo arms.
4. Resolve the per-satellite settings rule.
5. Release the dataset with a datasheet.

## 9. Atefrad & Karami checked against ESA's rules (added after reading the full paper)

The earlier table in section 3 used bootstrap means (.594, .765). The point estimates are .597 and .773, now corrected.

| Check | ESA expects | Atefrad & Karami | Ours (fleet) |
|---|---|---|---|
| Data | Real ESA telemetry | Real, M2 lightweight | **Simulated, ESA format** |
| Client | Not defined by ESA | Slices of one satellite's data | Separate satellites with own orbit, role, faults |
| Split | Test later than training | Stratified split, 50% overlapping windows; not chronological as described | Chronological |
| Streaming | R2: online, no future samples | Not shown | Past-only check passed, 36 cuts |
| Metric | Corrected event-wise F0.5 and others | Window-level P/R/F0.5, AUC, accuracy | ESA's own metric classes |
| Threshold-free metrics | Rejected (arXiv version) | AUC reported | Not used |
| Class balance | Natural | Oversampling, minimum anomalies per client | Natural |
| Rare events vs anomalies | Reported separately | Not separated | Both reported |
| Telecommands, non-target channels | R6 | Not used | Used |
| Affected channels | R5 | Not reported | Reported, weak (.49) |
| Centralised baseline | Not required | **Yes** | **No** |
| FedProx, 3-10 clients, comm cost | Not required | **Yes** | **No** (FedAvg, 4 clients, estimate only) |

Their numbers do not reconcile:

- Headline F0.5 0.86 with P .93, R .87. Those give .917; Table 6's confusion matrix gives .913.
- Table 7: recall 1.00 with F0.5 .87 implies precision about .84, and accuracy cannot then be below precision, yet accuracy is .77-.80.
- FL MLP: P .70, R 1.00, accuracy .71 implies about 68% anomalous test windows, against 4.29% stated and 9.5% in Table 6. It also resembles flagging nearly everything. Centralised accuracy is .97, so the two arms appear to be on different test balance.
- Nine of the 19 table rows checked miss the F0.5 formula by more than .015. Averaging across channels could explain some, not Table 6 or Table 7.
- Communication: 1.8 MB "per round" (abstract), "total" (section 5.3), 0.6-2.05 MB per round (Table 7).
- "FL beats centralised" compares fine-tuned FL with an untuned centralised model.

So 0.86 cannot be compared to any ESA-ADB score. It still counts as prior work for the idea of FedAvg + fine-tuning on ESA data, so cite it.

### Ours under ESA's own priority order

ESA's main table is all events, not anomalies only. Mean over four satellites:

| Arm | All-events F0.5 | Anomalies F0.5 | Channel-aware F0.5 | Subsystem F0.5 | Alarm precision | ADTQC |
|---|---|---|---|---|---|---|
| Solo, own settings | .754 | .393 | .443 | .451 | .763 | .861 |
| Solo, fleet settings | .785 | .589 | .435 | .433 | .763 | .934 |
| Plain FL, fleet settings | .752 | .597 | .617 | .699 | .749 | .903 |
| FL + fine-tune, own settings | .690 | .485 | .510 | .527 | .636 | .879 |
| FL + fine-tune, fleet settings | .902 | .773 | .487 | .487 | .807 | .928 |

- Fine-tuned FL with fleet settings beats solo on all four satellites on all events.
- Plain FL does not beat solo on all events (.752 vs .754), and SAT_4 falls from .788 to .418.
- Plain FL has the best channel and subsystem scores.

"As per ESA" should be worded as "evaluated with ESA-ADB's metrics and online rules on a simulated fleet". ESA defines no federated protocol and has not reviewed this. Our labels come from a twin-run 4-sigma rule, not operators. R5 and R7 are only partly met.

ESA journal version: same numbers as arXiv Table 2, but they sit in Tables 16-17, and its Table 2 is the requirements list. The "no telecommands in lightweight subsets" sentence and the rejection of threshold-free metrics were found in the arXiv version only.

### Prior work added

- Carneiro, Schaub et al. 2022: Basilisk wheel faults with ML, single spacecraft.
- Nadkarni et al.: Basilisk HR16 wheel friction faults with RUL (ARULE), single spacecraft.
- Mitra et al. 2026: federated turbofan RUL, client "health score" from fault counts, loss or validation F1, with failure-masking poisoning.

### Ideas that would make it stand out

1. Benchmark validity test: do algorithm rankings on the simulated fleet match ESA's published rankings? No study of this kind found. Needs no ESA data.
2. Map of when federation helps: sweep fleet size, fault overlap and unit variation, with centralised arm.
3. Sick-client study: does a degrading satellite damage fleet learning, and does gating updates with its own health index fix it? Links detection, FL and RUL. Related: Mitra et al.
4. Label-noise study: degrade exact labels to operator-style labels and measure how far ESA metrics move.
5. Fleet protection time: a fault on one satellite, an operator verdict, then time until the others are protected.
6. Simulation to real: does fleet pretraining help a real satellite with 14 days of data?

## Sources

- ESA-ADB: https://arxiv.org/abs/2406.17826 and DMLR version https://data.mlr.press/assets/pdf/v03-23.pdf
- Goetze et al.: https://arxiv.org/abs/2603.29375
- Allegrini & Pompei: https://arxiv.org/abs/2605.06681
- EVT: https://arxiv.org/abs/2609.10017
- Benchmark study: https://arxiv.org/abs/2607.07335
- Atefrad & Karami: https://publications.scrs.in/chapter/978-81-975670-0-1/3
- Laridi et al.: https://arxiv.org/html/2410.09284v1
- CNES/TeSA: https://hal.univ-grenoble-alpes.fr/IRIT-MINDS/hal-04156021v1
- Baireddy et al.: https://mlanthology.org/cvprw/2021/baireddy2021cvprw-spacecraft
- JPL feedback loop: https://gsaw.org/wp-content/uploads/2019/03/2019s05yun.pdf
- OPS-SAT-AD: https://www.nature.com/articles/s41597-025-05035-3
- Federated engine RUL: https://arxiv.org/abs/2506.00499
- Rudd repo: https://github.com/loganrudd/spacecraft-telemetry-anomaly-detection
- Sempere repo: https://github.com/JSempereH/ESA-ADB-anomaly-detection
- Atefrad & Karami full text: uploaded PDF, also https://repository.uel.ac.uk/item/8zyz2
- ESA journal version: https://data.mlr.press/assets/pdf/v03-23.pdf
- Carneiro et al.: https://hanspeterschaub.info/Papers/Carneiro2022.pdf
- Nadkarni et al.: https://www.ridgetopgroup.com/rgi_content/uploads/2025_/publications/Comprehensive%20Prognostic%20Evaluation%20of%20Spacecraft%20Reaction%20Wheels%20Post-Fault%20Occurrences.pdf
- Mitra et al.: https://arxiv.org/abs/2608.04045
