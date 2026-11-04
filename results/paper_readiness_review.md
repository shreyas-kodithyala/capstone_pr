# Paper readiness review: Basilisk fleet, federated learning, RUL

Date: 4 Nov 2026. Scope: final fresh-seed fleet (`sim/fleet_final.json`), the code in this repository, and a literature check.
Evidence: `results/audit/` (scripts and raw output). Numbers below were measured in this review unless marked otherwise.

## 1. Verdict

| Question | Answer |
|---|---|
| Is the work correct? | **Yes on integrity.** No leakage, causal detector, metric reproduced by hand. Two bugs found and fixed before scoring. |
| Is the dataset realistic? | **Physically yes; statistically it looks like a different mission, not like ESA.** Anomaly types, density and difficulty differ. |
| Does it solve a real problem? | **Partly.** False alarms and scarce fault data are real. The case for federating rather than centralising is not yet made. |
| Is it novel? | **The FL method is not. The dataset probably is.** No "first" claim is safe. |
| Top-tier ready? | **No.** Single seed and one fleet are decisive. Good capstone; publishable at a domain venue after the fixes in section 7. |

## 2. Dataset realism

| Check | Standard | Result | Verdict |
|---|---|---|---|
| Momentum conservation | physics | drift 2.2e-10 | Pass |
| Eclipse duration | analytic model | within 1.3-1.5 min | Pass |
| Orbit rate | analytic n = 0.0627 deg/s | 0.060-0.064 (gyro bias) | Pass |
| Actuator limits | HR16 datasheet | under 3,755 rpm of 6,000; current at the 2 A limit plus noise | Pass |
| Battery | 8S Li-ion | discharges in 99.7 %+ of eclipses, 30.8-33.7 V | Pass (shallow cycling) |
| ESA format | ESA-ADB layout and metrics | pipeline runs unchanged | Pass |
| Label quality | Wu & Keogh mislabelling flaw | exact, from a fault-free twin (4 sigma) | **Strength** |
| Anomalous time | ESA 0.77-1.57 % | 0.07-1.80 %; SAT_2 test 38.9 %, 1.19 % without run-to-failure | Pass, one exception |
| Anomaly rate | ESA 3.7-5.4 per year | 10-43 per satellite-year | Deliberately denser (about 2-12x) |
| Rare events | ESA 4.6 (M1), 173 (M2) per year | 9-28 per year | Pass |
| Anomaly type mix | ESA M1 94 % multivariate-global; M2 47 % global, 47 % local | ours 58 % multivariate-local, 16 % multivariate-global | **Fail vs M1**, partial vs M2 |
| Point anomalies | ESA M1 21.6 %, M2 0 % | 2.3 % | Partial |
| catch22 coverage | Lubba et al. 2019 | median 0.99; nearest-neighbour closeness 0.77 (real 0.95) | Pass |
| catch22 distributions | KS per feature | median 0.31; 3 of 22 below 0.2 | Weak |
| Classifier two-sample test | Lopez-Paz & Oquab 2017 | AUC 1.000 vs ESA; 0.994-1.000 per group | **Inconclusive**: ESA M1 vs M2 is also 1.000 |
| Triviality | Wu & Keogh one-liner | 43 of 59 events (73 %); 26 of 39 (67 %) without wear events | **Weak** |
| Limit-check baseline | ESA practice | anomaly recall 0.12-0.40, F0.5 0.37-0.59 | Non-trivial for real limits |
| Closed-loop sensors | realism | `SimpleNav` has no noise: sensor faults never reach the control loop | **Weak** |
| Scale | ESA 76-100 channels, 123-698 commands, 3.5-14 years | 20 channels, 12 commands, 1 year | Weak |

Notes:
- The one-liner test uses hindsight: it picks the best channel and threshold per event. It still flags that most faults are large on one channel. Our strengthened faults add to this (C3 is 5 of 5).
- The two-sample test separates real missions from each other, so it says we are a different mission, not that we are unrealistic. No test here shows we are "like ESA" beyond coverage and label statistics.
- Correction: earlier messages said anomalies are "mostly local, as on ESA", and `HANDOFF.md` repeats it. That is wrong. The ESA paper states most annotations are multivariate global.

**How to describe it:** a physics-grounded simulated fleet in ESA format with exact labels, whose time share of anomalies matches ESA. Not "ESA-like" in general.

## 3. Correctness audit

| Check | Result |
|---|---|
| Labels as model input | None. Labels only mask the loss |
| Training rows | end at row 335,374; early stopping ends at 394,559; validation starts 394,560; test starts 524,160 |
| Normalisation | rows before validation only |
| Model causality | changing future inputs changed earlier errors by exactly 0.0 |
| Detector past-only | passed for solo local, fed fleet, pers fleet: 36 cuts, 0 differences before the 70-row limit |
| ESA precision by hand | 0.832, 0.921, 0.888, 0.999 from counts, all equal to the metric |
| Generator code | unchanged since the final seeds were drawn |
| FedAvg | shared layers only, embeddings local. Plain mean: **no robust filtering is implemented** |

Problems found:
1. **Look-ahead rule break (fixed):** alarm merging of up to 2,880 rows in rounds 3-5. Those reports are invalid and marked so. Round 5b replaces them.
2. **Metric crash on zero alarms (fixed):** before any fresh-seed detection score was produced.
3. **Headline changed after scoring (open):** see section 5.
4. **Weak solo baseline:** solo with local selection (mean 0.393) is worse than solo with no tuning (0.535 at k3). SAT_1's 0/6 is a local-selection failure, and it inflates the gain.
5. **Small unfairness:** pers gets 1,900 gradient steps against solo's 1,800.
6. **Detector designed on the development fleet.** The final fleet is clean of that, but RUL's threshold rule was picked after seeing one early-bias result on SAT_2 (rule chosen on archive wheels only).

## 4. RUL

Result on fresh seeds: 100-day alert on the correct wheel, MAE 2.2 days, alpha-lambda 0.83, 15 % late.

**It is partly circular, and a reviewer will see it:**
- every wheel fails at the same wear value (`WEAR_W fail=79`);
- every wheel follows the same exponential mean path, and the predictor fits an exponential;
- it is always wheel `rw2`, and the archive wheels come from the same generator (HI at failure 61.0-65.3).

So the method mostly recovers the generator's assumptions. The straight-line failure (MAE 271) shows only that the generator is exponential. Real bearing failures vary. Claim the pipeline, not the accuracy.

## 5. Model and FL results

Macro mean anomaly F0.5 on the final fleet. Rows other than the three reported arms are post-hoc analysis: they changed nothing.

| Configuration | Mean F0.5 | Caught of 59 | False alarms |
|---|---|---|---|
| solo, local selection | 0.393 | 42 | 25 |
| solo, no tuning (k3) | 0.535 | 48 | 91 |
| solo, fleet selection | 0.589 | 37 | 21 |
| fed, fleet selection (freeze headline) | 0.597 | 44 | 32 |
| pers, local selection | 0.486 | 50 | 71 |
| pers, no tuning (k3) | 0.637 | 48 | 62 |
| **pers, fleet selection (original headline)** | **0.773** | **44** | **4** |
| pers, own ceilings instead of fleet | 0.684 | 49 | 22 |
| pers, settings chosen without own satellite | **0.776** | 41 | 3 |
| pers, no command mask | 0.454 | 9 | 1 |
| pers, one absolute threshold (a30) | 0.228 | 38 | 919 |

What this shows:
- **Calibration matters more than federation.** Per-channel thresholds and the command mask are the largest effects.
- **Leave-one-satellite-out works (0.776).** Settings chosen without a satellite's data transfer to it. This is the most interesting FL result.
- **Sharing noise ceilings cuts false alarms 22 to 4**, at a cost of 5 catches. SAT_3 gets worse with it (0.952 to 0.784).
- **Weights help at equal settings:** at pers's fleet setting, 0.773 against solo's 0.627. Better in 9 of 12 same-setting comparisons, not all.
- **The gain is mostly precision.** Catches barely move; false alarms fall.

Statistics (bootstrap over test events, false alarms Poisson):

| Arm | Mean F0.5 (95 % interval) | Gain over solo local |
|---|---|---|
| solo local | 0.393 (0.312-0.500) | - |
| fed fleet | 0.594 (0.448-0.692) | +0.198 (0.030-0.336) |
| pers fleet | 0.765 (0.578-0.884) | +0.353 (0.165-0.520) |

These intervals cover event sampling only. They do **not** cover training seeds, fleets or selection, and those are larger: SAT_1 pers went 0.333 to 0.832 between fleets, and SAT_4 fed went 0.586 to 0.082. The mean gain was positive on both fleets (the development fleet is biased), which is encouraging but is n = 2.

**Headline issue.** The design freeze named fed fleet (3 of 4 satellites improved; SAT_4 fell to 0.082). The original design named pers fleet (4 of 4). Using pers is a post-hoc choice and must be disclosed.

Weak spots to state:
- Component attribution is poor: family accuracy 0.60, 0.77, 0.50, 0.00; channel-aware F0.5 0.15-0.41. The problem statement says the system "flags the affected component".
- SAT_3 catches fall from 15/15 to 8/15. SAT_4 catches 1/5 and limit checks beat it (0.588 vs 0.555).

## 6. Problem and novelty

**The problem is real, with one gap.** ESA says false positives are costly and that common detectors are not yet deployable, and scarce fault examples per spacecraft are a genuine limit. The gap is "why not centralise?". One operator owns the fleet, and housekeeping telemetry is routinely downlinked. FL here moves about 19 MB per satellite (0.81 MB x 2 directions x 12 rounds) against about 32 MB of raw telemetry for the 137-day window, so bandwidth barely favours it. There is no centralised upper bound, though `REPORT.md` planned one. Stronger reasons exist (multi-operator fleets, onboard autonomy, ISL-only operation) but none is tested.

| Claim | Closest prior work | Status |
|---|---|---|
| FL for satellite telemetry anomaly detection | Atefrad & Karami 2025 (FedAvg + fine-tuning on ESA-ADB M2, F0.5 0.86); STA-FedAvg; attention LSTM-AE | **Not new** |
| Fleet picks a threshold by averaged client score | Laridi et al. 2024: the same mechanism with F1 | **Not new** |
| Shared layers, local personal parts | PFAE; FedPer-style partial sharing (from memory) | **Not new** |
| Sharing per-channel noise ceilings | Laridi shares summary statistics, differently | Small, possibly new |
| Command mask | no paper found; mode-dependent limits are standard practice | Useful engineering, not novelty |
| Telecommand-attention graph forecaster with availability features on ESA-ADB | nothing found in two searches | Possibly new; needs a wider search |
| Public physics-simulated multi-satellite fault benchmark, ESA format, exact labels, run-to-failure | nothing found; nearest are OPS-SAT-AD, ESA-AD (real, single mission) and a DLR synthetic generator | **Strongest candidate** |
| Fleet-informed RUL | nothing found at fleet level | Open, but circular today |

The searches were limited, so write "we are not aware of", never "first".

**Real-data claims need correcting (`CLAUDE.md`, `REPORT.md`):**
- 0.786 and 0.949 are confirmed as ESA's published best all-events scores on the lightweight subsets.
- "Beats every published result" is not true as stated. ESA's Telemanom-ESA-Pruned scores **0.968 anomalies-only on M1; ours is 0.939**.
- ESA's lightweight subsets use no telecommands; GDN2 used 30 (M1) and 123 (M2). Availability features also come from raw files. Comparison is not like-for-like.
- Goetze et al. 2026 report 92.7 corrected event-wise F0.5 on M1 lightweight, with an unclear test protocol. It cannot be ignored.
- Not re-verified here: ESA data is not on this machine.

## 7. What would make it stand out

**Best paper shape:** an open fleet benchmark plus a careful study of when federation helps. Lead with the dataset, and with the findings that calibration beats federation, and that fleet settings transfer to a satellite that contributed nothing.

| Priority | Fix | Estimated effort |
|---|---|---|
| 1 | 5 fresh fleets x 3 training seeds, intervals across them | about 14 h on this machine, runs unattended |
| 2 | Centralised upper bound; FedProx and FedPer-style arms; untuned solo as the baseline; Telemanom and DC-VAE on the fleet | 3 h compute plus coding |
| 3 | RUL: random failure level per wheel, random wheel, several degradation shapes, more failures | half a day plus regeneration |
| 4 | Closed-loop sensors (noisy navigation, gyro faults reach the controller) | 1-2 days |
| 5 | Harder, more global faults; aim for under 40 % one-liner, closer to ESA's type mix | 1 day plus regeneration |
| 6 | ESA rerun with no telecommands, and compare with Goetze et al. | needs ESA data |
| 7 | Communication analysis and a robust-aggregation test; show one case where centralising fails | 1 day |
| 8 | Release dataset and generator with a datasheet | 1 day |

**Venue, as judgement:** not NeurIPS, ICML or KDD main track as it stands. After items 1-5 and 8, NeurIPS Datasets & Benchmarks becomes a stretch target. After items 1-3, IEEE Aerospace, AIAA SciTech, SmallSat, Acta Astronautica or PHM Society are realistic.

## 8. Sources

- Kotowski et al., ESA-ADB: https://arxiv.org/pdf/2406.17826
- Atefrad & Karami 2025: https://www.publications.scrs.in/uploads/final_menuscript/aa719ab2744a80d718f57bfaaffccea8.pdf
- Laridi et al. 2024: https://arxiv.org/html/2410.09284v1
- Goetze et al. 2026: https://arxiv.org/html/2603.29375
- Benchmark study 2026: https://arxiv.org/html/2607.07335v1
- STA-FedAvg: https://www.researchgate.net/publication/410625899
- Attention LSTM-AE FL: https://www.researchgate.net/publication/414822873
- OPS-SAT-AD: https://www.nature.com/articles/s41597-025-05035-3
- DLR synthetic telemetry: https://elib.dlr.de/222516/1/s12567-024-00589-1.pdf
- On-orbit AI survey: https://arxiv.org/pdf/2604.16518
- Howard & Ayoubi (RUL, synthetic): https://www.citedrive.com/en/discovery/autonomous-fault-detection-and-prognosis-for-spacecraft-reaction-wheels-using-sparse-identification/
- ESA-ADB leaderboard: https://www.sota2.com/research/sota/anomaly-detection-on-esa-adb
