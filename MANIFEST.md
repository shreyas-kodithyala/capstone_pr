# Manifest: where everything is

Start with `CLAUDE.md`, then `FULL_PROJECT_REPORT.md`, then `HANDOFF.md`.

The work is stored in two places:

| Place | Contents | Size |
|---|---|---|
| This git repository | Code, reports, results, logs, labels, package lists | About 6 MB |
| GitHub Release `v1.0-data` | Full datasets, physics records, models, forecast errors | 15.7 GB in 8 pieces |

A `git clone` does not download Release files. Download them separately.

## In the repository

| Path | Contents |
|---|---|
| `CLAUDE.md` | Project rules. Some result claims are out of date; see report section 16 |
| `FULL_PROJECT_REPORT.md` | Complete step-by-step report |
| `HANDOFF.md` | Dated work log and every development round |
| `REPORT.md` | ESA-phase report and original plan |
| `MANIFEST.md` | This file |
| `sim/` | Dataset generator, validation, baselines, final seeds (`fleet_final.json`), ESA reference statistics |
| `fed/` | Federated learning, RUL, causality check, final pipeline (`run_final.sh`) |
| `ext/new_runs/` | Frozen GDN2 model, detector stack and ESA metric code |
| `envs/` | Original environment files, plus exact package lists from this machine (`localenv_*`) |
| `results/final/` | Final results, logs, `all_metrics.md`, `per_event_detection.csv`, RUL output |
| `results/rounds/` | Reports from the development rounds |
| `results/audit/` | Audit scripts, outputs, the causality log and the piece check |
| `results/audit/scratch/` | Small diagnostic scripts and outputs |
| `results/dataset_meta/` | Labels, event truth, channel lists, wear and health truth for every satellite and archive wheel |
| `results/logs/` | Dataset generation and environment setup logs |
| `results/paper_readiness_review.md` | Paper readiness review |
| `results/project_synthesis_review.md` | Whole-project review and related work |
| `results/release_SHA256SUMS.txt` | Fingerprints of the 8 Release pieces |

## In the Release

The 8 pieces join to make one compressed archive:

| Folder in the archive | Size | Contents |
|---|---|---|
| `data/sim_fleet/` | 1.2 GB | **Final dataset** in ESA format, including the archive wheels and validation output |
| `data/sim_work/` | 6.6 GB | Final raw Basilisk records and fault-free twins |
| `data/sim_fleet_v2c/` | 1.2 GB | Development dataset |
| `data/sim_work_v2c/` | 6.6 GB | Development raw records |
| `data/*.log` | Small | Generation and setup logs |
| `fed/out_final/` | 1.5 GB | **Final run:** models, forecast errors, input cache, results |
| `fed/out_v2/` | 1.5 GB | Development rounds |
| `fed/out_w60/` | 1.5 GB | 60-day window test |

Total: 1,176 files and 21.6 GB before compression.

| Piece | SHA-256 |
|---|---|
| `capstone_data.tar.gz.part-aa` | `4fc3d5d58e059266528cee4f986db6c75f1a64a7730a98985aad160cd98e98f5` |
| `capstone_data.tar.gz.part-ab` | `33c6bc59a8a33d5c6e0be7243f8e4b0e69ebeb7c7abfac81df48a83736aabf39` |
| `capstone_data.tar.gz.part-ac` | `f6cceea2b934be2a16e8918abac387a2745c094a5fc96c3ca5b32ac0cc5a693b` |
| `capstone_data.tar.gz.part-ad` | `e358c8f0b869be172a0c1db474bf686f4d80f50f6a765d7c2cd61c3950f32514` |
| `capstone_data.tar.gz.part-ae` | `fb95b42ecc8bec45a099ca738cf0acb342e455f9110c7977297ee57746a91de2` |
| `capstone_data.tar.gz.part-af` | `6afd74fb67552d1a38e7a878e187ffa895074a02ea27a7d50f1dddfd81aff0ce` |
| `capstone_data.tar.gz.part-ag` | `fac274248b04534d3b267bf1cb95a74c2ded7758f7a7c7d546bc01efe441328c` |
| `capstone_data.tar.gz.part-ah` | `0fbcc8624cea6568acdd9f6217effabc1ba9d5a838efb05db1b999f33f366e3e` |

## Restore the big files

Run these inside a clone of the repository:

```bash
gh release download v1.0-data --repo shreyas-kodithyala/capstone_pr
shasum -a 256 -c SHA256SUMS.txt
cat capstone_data.tar.gz.part-* | tar -xzf -
```

On Linux, `sha256sum -c SHA256SUMS.txt` replaces the `shasum` line. Every line must say `OK` before unpacking.

The archive also holds 1,300 small macOS `._*` metadata entries. macOS handles them. On Linux they appear as extra files and are safe to delete:

```bash
find data fed -name '._*' -delete
```

After restoring, `data/` and `fed/out_*` stay ignored by git.

## Verified on 5 Nov 2026

- All 8 pieces matched `SHA256SUMS.txt`.
- All 1,176 files were read back from the pieces and matched the originals byte for byte.
- No file was missing and no file differed.

Evidence: `results/audit/verify_release.py` and `results/audit/verify_release.log`. The log lists the `._*` entries as `missing_on_disk`, which is expected. It prints only the first 50 of them.

## Rebuild the environments

The local toolchain is not uploaded. Use `HANDOFF.md` section 6.2, with exact versions from:

- `envs/localenv_bsk_pip.txt` and `envs/localenv_bsk_conda.txt` (Python 3.11, Basilisk 2.12.2);
- `envs/localenv_timeeval_pip.txt` and `envs/localenv_timeeval_conda.txt` (Python 3.9, PyTorch);
- `envs/localenv_activate.sh`, which shows how this machine kept tools inside the folder. Its paths are specific to this machine.

## Not included

| Item | Why | Replacement |
|---|---|---|
| `.localenv/miniforge`, `.localenv/cache`, installer | Large and machine-specific | Package lists in `envs/` |
| `.localenv/tmp/papers`, `esa.txt`, `adaptive.txt` | Third-party paper text | Original PDFs, and citations in the reviews |
| `.localenv/tmp/sim_c22.npy` | Small audit array | Rerun `results/audit/audit_c.py` |
| `__pycache__/` | Generated files | Recreated automatically |
| ESA-ADB data and `ESA/Main/` | Never in this folder | Original source |
| `data/sim_fleet_v1/`, `fed/out_v2_r1/` | Stayed on the old Mac, not needed | Described in `HANDOFF.md` |
| Full chat transcript | Stored outside this folder | Context is in report sections 19–20 and `HANDOFF.md` section 9 |

## Privacy note

Logs and `envs/localenv_activate.sh` contain this machine's local folder paths, including its username. If the repository stays public, those paths are public too.
