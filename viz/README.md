# Viewing the simulation

There are two ways to show the fleet. Use both for the panel.

| Tool | Shows | Needs |
|---|---|---|
| **Vizard** | The satellites in 3D: orbits, attitude, wheels, Sun, Earth, ground stations | The free Vizard app and a `.bin` file |
| **Mission Replay** (`mission_replay.html`) | The whole year: every telecommand, telemetry, every event, degradation, failure, RUL and scores | A web browser, no internet |

Everything here replays the saved final-fleet records. Nothing is simulated again.

## Open Mission Replay

Double-click `viz/mission_replay.html`, then press **Play**. Use **Jump to** for key moments, such as the wheel failure.

## Open a file in Vizard

1. Install Vizard on the other laptop from the [Vizard download page](https://avslab.github.io/basilisk/develop/Vizard/VizardDownload.html). Start with the latest release.
2. Open Vizard.
3. Choose **Basilisk Message File**, then **Select**, and pick a `.bin` file.
4. Press **Start Visualization**.

**Test before the panel.** Start with `clip8_SAT_2_wheel_failure.bin`, because it's small. These files were checked by decoding them, but they have not been opened in Vizard yet, so the pop-ups and colours still need checking there. If Vizard can't read a file, a different Vizard version is the first thing to try.

## Short clips: full 5-second detail

These are in `viz/clips/`, in git. Every telecommand appears as a pop-up.

| File | Satellites | Shows | Mission days |
|---|---|---|---|
| `clip1_fleet_day200.bin` | All 4 | Normal fleet operations for 6 hours | 200.00 to 200.25 |
| `clip2_SAT_4_imaging.bin` | `SAT_4` | Imaging slews and payload use | 250.00 to 250.13 |
| `clip3_SAT_4_gyro_calibration.bin` | `SAT_4` | Gyro calibration manoeuvres | 225.38 to 225.46 |
| `clip4_SAT_4_safe_drill.bin` | `SAT_4` | Safe-mode drill | 246.57 to 246.77 |
| `clip5_SAT_1_friction_spike.bin` | `SAT_1` | Wheel 4 friction rises 10.5 times | 204.74 to 204.91 |
| `clip6_SAT_3_wheel_dropout.bin` | `SAT_3` | Wheel 3 drive drops out for about 2 minutes | 267.53 to 267.59 |
| `clip7_SAT_2_wear_spike.bin` | `SAT_2` | Wear friction spike on wheel 2 | 286.84 to 286.98 |
| `clip8_SAT_2_wheel_failure.bin` | `SAT_2` | Wheel 2 fails at day 356.11; the satellite continues on 3 wheels | 356.02 to 356.27 |

## Whole-year files

These are too large for git, so they go in GitHub Release `v1.1-vizard`. Download them into `viz/large/`.

| File | Satellites | Step | Size | Telecommand pop-ups |
|---|---|---|---|---|
| `fleet_year_60s.bin` | All 4, all 365 days | 60 s | 1.1 GB | No, events only |
| `fleet_Q1_60s.bin` to `fleet_Q4_60s.bin` | All 4, one quarter each | 60 s | 285 MB each | No, events only |
| `SAT_1_year_30s.bin` to `SAT_4_year_30s.bin` | One satellite, all 365 days | 30 s | 840 MB each | Yes, every command |

If the full-year file is slow to open, use a quarter file. `fleet_Q4_60s.bin` covers days 273 to 365, including the degradation and failure.

## Colour key

The colour of each satellite's trajectory line shows its state.

| Colour | Meaning |
|---|---|
| Blue | Normal |
| Aqua | Slewing |
| Yellow | Rare event |
| Red | Anomaly |
| Orange | Degrading |
| Purple | Wheel failed |

Wheel speeds and torques are in Vizard's reaction wheel panel. It's switched on for each satellite in these files.

## Limits

- **Whole-year detail:** the record is at 5 seconds, but the whole-year files use 30 or 60 seconds. All 5-second frames would be about 14 GB for the fleet. The short clips keep full detail.
- **Watching a full year:** even at high speed this takes a long time, so jump along the timeline. Mission Replay plays the year in about 6 minutes at 1 day per second.
- **Sensor, power and thermal faults:** these don't change the satellite's motion, because navigation is perfect in the simulator. In Vizard they appear only as a colour and pop-up. See the telemetry in Mission Replay.
- **Satellite model:** the 3D shape is Vizard's generic satellite, not a real spacecraft design.
- **Telemetry in Mission Replay:** it uses 3-hour average, low and high values from the 30-second dataset.
- **False alarms:** their times were not stored, so they appear only as final totals.
- **Labels:** colours and pop-ups use ground-truth labels for display only. The models never received them.
- **Timing:** training and FL ran offline, not live in orbit.

## How the files were made and checked

- `sim/viz_replay.py` builds the Vizard files from `data/sim_work/`. Run `python viz_replay.py --list` to see the names.
- `sim/viz_dashboard.py` builds Mission Replay.
- `sim/viz_check.py` decodes each `.bin` file and compares sampled frames with the saved records. Results are in `viz/verification.json`.

Each satellite's saved record uses its own slowly turning frame. The replay rotates all of them into one shared frame, using the same rotation the simulator used for Sun, eclipse and ground-pass geometry.

Fingerprints of the whole-year files are in `viz/large_SHA256SUMS.txt`.
