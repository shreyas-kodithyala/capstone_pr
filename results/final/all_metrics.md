# All final-fleet metrics

Built by `make_all_metrics.py` from `results.json` and `matched_*.json`. Test days 182-364, scored once, one seed.

## Fleet means: All events (anomalies + rare events)

| Arm | Settings | F0.5 | Precision | Recall | Alarm precision | AFF F0.5 | Channel F0.5 | Subsystem F0.5 | ADTQC |
|---|---|---|---|---|---|---|---|---|---|
| Solo | own settings | 0.754 | 0.792 | 0.750 | 0.763 | 0.792 | 0.443 | 0.451 | 0.861 |
| Solo | fleet settings | 0.785 | 0.819 | 0.707 | 0.763 | 0.803 | 0.435 | 0.433 | 0.934 |
| Plain FL | own settings | 0.689 | 0.707 | 0.816 | 0.697 | 0.839 | 0.521 | 0.649 | 0.863 |
| Plain FL | fleet settings | 0.752 | 0.778 | 0.758 | 0.749 | 0.844 | 0.617 | 0.699 | 0.903 |
| FL + personalization | own settings | 0.690 | 0.681 | 0.845 | 0.636 | 0.849 | 0.510 | 0.527 | 0.879 |
| FL + personalization | fleet settings | 0.902 | 0.940 | 0.788 | 0.807 | 0.875 | 0.487 | 0.487 | 0.928 |

## Fleet means: Anomalies only

| Arm | Settings | F0.5 | Precision | Recall | Alarm precision | AFF F0.5 | Channel F0.5 | Subsystem F0.5 | ADTQC |
|---|---|---|---|---|---|---|---|---|---|
| Solo | own settings | 0.393 | 0.374 | 0.497 | 0.399 | 0.512 | 0.157 | 0.238 | 0.833 |
| Solo | fleet settings | 0.589 | 0.646 | 0.489 | 0.649 | 0.636 | 0.153 | 0.253 | 0.949 |
| Plain FL | own settings | 0.533 | 0.640 | 0.660 | 0.592 | 0.714 | 0.315 | 0.491 | 0.891 |
| Plain FL | fleet settings | 0.597 | 0.647 | 0.568 | 0.649 | 0.712 | 0.416 | 0.542 | 0.877 |
| FL + personalization | own settings | 0.485 | 0.480 | 0.660 | 0.466 | 0.701 | 0.262 | 0.376 | 0.895 |
| FL + personalization | fleet settings | 0.773 | 0.910 | 0.619 | 0.690 | 0.751 | 0.240 | 0.352 | 0.910 |

## Fleet totals

| Arm | Settings | Caught of 59 | False alarms | Rare events caught of 30 | Unseen caught of 27 | Seen caught of 32 |
|---|---|---|---|---|---|---|
| Solo | own settings | 42 | 25 | 28 | 15 | 27 |
| Solo | fleet settings | 37 | 21 | 28 | 13 | 24 |
| Plain FL | own settings | 49 | 58 | 28 | 20 | 29 |
| Plain FL | fleet settings | 44 | 32 | 28 | 18 | 26 |
| FL + personalization | own settings | 50 | 71 | 28 | 19 | 31 |
| FL + personalization | fleet settings | 44 | 4 | 28 | 18 | 26 |

## Per satellite

| Satellite | Arm | Settings | F0.5 all | F0.5 anomalies | Precision anom | Recall anom | Channel F0.5 all | Subsystem F0.5 all | Alarm precision all | ADTQC all | Caught | False alarms | False-alarm time % | Rare caught | Median delay min | Family attribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SAT_1 | Solo | own settings | 0.781 | 0.000 | 0.000 | 0.000 | 0.348 | 0.347 | 1.000 | 1.000 | 0/6 | 0 | 0.02 | 5/6 | - | - |
| SAT_1 | Solo | fleet settings | 0.909 | 0.833 | 0.999 | 0.500 | 0.419 | 0.421 | 1.000 | 1.000 | 3/6 | 0 | 0.07 | 5/6 | 0.5 | 0.33 |
| SAT_1 | Plain FL | own settings | 0.833 | 0.500 | 1.000 | 0.167 | 0.368 | 0.330 | 1.000 | 1.000 | 1/6 | 0 | 0.02 | 5/6 | 18.0 | 0.00 |
| SAT_1 | Plain FL | fleet settings | 0.795 | 0.555 | 0.666 | 0.333 | 0.528 | 0.514 | 1.000 | 1.000 | 2/6 | 1 | 0.06 | 5/6 | 4.0 | 1.00 |
| SAT_1 | FL + personalization | own settings | 0.833 | 0.682 | 0.750 | 0.500 | 0.570 | 0.583 | 0.500 | 0.999 | 3/6 | 1 | 0.03 | 5/6 | 21.5 | 1.00 |
| SAT_1 | FL + personalization | fleet settings | 0.892 | 0.832 | 0.832 | 0.833 | 0.544 | 0.588 | 1.000 | 0.892 | 5/6 | 1 | 0.13 | 5/6 | 1.0 | 0.60 |
| SAT_2 | Solo | own settings | 0.670 | 0.636 | 0.607 | 0.788 | 0.429 | 0.387 | 0.113 | 0.745 | 26/33 | 15 | 4.28 | 5/6 | 0.0 | 0.81 |
| SAT_2 | Solo | fleet settings | 0.670 | 0.636 | 0.607 | 0.788 | 0.429 | 0.387 | 0.113 | 0.745 | 26/33 | 15 | 4.28 | 5/6 | 0.0 | 0.81 |
| SAT_2 | Plain FL | own settings | 0.878 | 0.870 | 0.854 | 0.939 | 0.584 | 0.896 | 0.108 | 0.770 | 31/33 | 4 | 3.54 | 5/6 | 1.0 | 0.90 |
| SAT_2 | Plain FL | fleet settings | 0.878 | 0.870 | 0.854 | 0.939 | 0.584 | 0.896 | 0.108 | 0.770 | 31/33 | 4 | 3.54 | 5/6 | 1.0 | 0.90 |
| SAT_2 | FL + personalization | own settings | 0.397 | 0.365 | 0.316 | 0.939 | 0.466 | 0.420 | 0.106 | 0.769 | 31/33 | 58 | 9.15 | 5/6 | 0.0 | 0.77 |
| SAT_2 | FL + personalization | fleet settings | 0.923 | 0.919 | 0.921 | 0.909 | 0.449 | 0.411 | 0.292 | 0.834 | 30/33 | 2 | 1.73 | 5/6 | 0.0 | 0.77 |
| SAT_3 | Solo | own settings | 0.776 | 0.736 | 0.691 | 1.000 | 0.346 | 0.506 | 1.000 | 0.697 | 15/15 | 6 | 3.29 | 4/4 | 0.0 | 0.27 |
| SAT_3 | Solo | fleet settings | 0.774 | 0.686 | 0.777 | 0.467 | 0.243 | 0.359 | 1.000 | 0.990 | 7/15 | 2 | 0.08 | 4/4 | 1.5 | 0.29 |
| SAT_3 | Plain FL | own settings | 0.736 | 0.688 | 0.646 | 0.933 | 0.341 | 0.501 | 1.000 | 0.683 | 14/15 | 7 | 3.16 | 4/4 | 1.2 | 0.29 |
| SAT_3 | Plain FL | fleet settings | 0.915 | 0.882 | 1.000 | 0.600 | 0.599 | 0.632 | 1.000 | 0.857 | 9/15 | 0 | 0.03 | 4/4 | 17.5 | 1.00 |
| SAT_3 | FL + personalization | own settings | 0.705 | 0.657 | 0.605 | 1.000 | 0.356 | 0.538 | 1.000 | 0.750 | 15/15 | 9 | 3.18 | 4/4 | 0.0 | 0.27 |
| SAT_3 | FL + personalization | fleet settings | 0.845 | 0.784 | 0.888 | 0.533 | 0.302 | 0.386 | 1.000 | 0.989 | 8/15 | 1 | 0.05 | 4/4 | 0.5 | 0.50 |
| SAT_4 | Solo | own settings | 0.788 | 0.200 | 0.199 | 0.200 | 0.651 | 0.564 | 0.938 | 1.000 | 1/5 | 4 | 0.25 | 14/14 | 6.0 | 0.00 |
| SAT_4 | Solo | fleet settings | 0.788 | 0.200 | 0.199 | 0.200 | 0.651 | 0.564 | 0.938 | 1.000 | 1/5 | 4 | 0.25 | 14/14 | 6.0 | 0.00 |
| SAT_4 | Plain FL | own settings | 0.307 | 0.073 | 0.060 | 0.600 | 0.792 | 0.868 | 0.680 | 1.000 | 3/5 | 47 | 0.69 | 14/14 | 0.0 | 1.00 |
| SAT_4 | Plain FL | fleet settings | 0.418 | 0.082 | 0.069 | 0.400 | 0.759 | 0.754 | 0.889 | 0.986 | 2/5 | 27 | 0.22 | 14/14 | 40.5 | 1.00 |
| SAT_4 | FL + personalization | own settings | 0.823 | 0.238 | 0.250 | 0.200 | 0.651 | 0.564 | 0.938 | 1.000 | 1/5 | 3 | 0.19 | 14/14 | 7.0 | 0.00 |
| SAT_4 | FL + personalization | fleet settings | 0.948 | 0.555 | 0.999 | 0.200 | 0.651 | 0.564 | 0.938 | 1.000 | 1/5 | 0 | 0.13 | 14/14 | 11.0 | 0.00 |

## Recall by fault family

| Satellite | Trained on | Arm | Settings | F1 wheel | F2 sensor | F3 power/thermal |
|---|---|---|---|---|---|---|
| SAT_1 | F2 | Solo | own settings | 0.00 | 0.00 | 0.00 |
| SAT_1 | F2 | Solo | fleet settings | 0.50 | 1.00 | 0.33 |
| SAT_1 | F2 | Plain FL | own settings | 0.00 | 1.00 | 0.00 |
| SAT_1 | F2 | Plain FL | fleet settings | 0.50 | 1.00 | 0.00 |
| SAT_1 | F2 | FL + personalization | own settings | 0.50 | 1.00 | 0.33 |
| SAT_1 | F2 | FL + personalization | fleet settings | 1.00 | 1.00 | 0.67 |
| SAT_2 | F1 | Solo | own settings | 0.84 | 0.62 | - |
| SAT_2 | F1 | Solo | fleet settings | 0.84 | 0.62 | - |
| SAT_2 | F1 | Plain FL | own settings | 0.92 | 1.00 | - |
| SAT_2 | F1 | Plain FL | fleet settings | 0.92 | 1.00 | - |
| SAT_2 | F1 | FL + personalization | own settings | 0.96 | 0.88 | - |
| SAT_2 | F1 | FL + personalization | fleet settings | 0.92 | 0.88 | - |
| SAT_3 | F3 | Solo | own settings | 1.00 | 1.00 | 1.00 |
| SAT_3 | F3 | Solo | fleet settings | 0.50 | 0.60 | 0.33 |
| SAT_3 | F3 | Plain FL | own settings | 1.00 | 1.00 | 0.83 |
| SAT_3 | F3 | Plain FL | fleet settings | 0.50 | 1.00 | 0.33 |
| SAT_3 | F3 | FL + personalization | own settings | 1.00 | 1.00 | 1.00 |
| SAT_3 | F3 | FL + personalization | fleet settings | 0.75 | 0.60 | 0.33 |
| SAT_4 | F1 F2 | Solo | own settings | - | - | 0.20 |
| SAT_4 | F1 F2 | Solo | fleet settings | - | - | 0.20 |
| SAT_4 | F1 F2 | Plain FL | own settings | - | - | 0.60 |
| SAT_4 | F1 F2 | Plain FL | fleet settings | - | - | 0.40 |
| SAT_4 | F1 F2 | FL + personalization | own settings | - | - | 0.20 |
| SAT_4 | F1 F2 | FL + personalization | fleet settings | - | - | 0.20 |

## Same settings for all models (matched)

Each row uses the fleet setting chosen by the named arm. Cells: F0.5 all / F0.5 anomalies / caught / false alarms.

| Setting from | Satellite | Solo model | Plain FL model | FL + personalization model |
|---|---|---|---|---|
| Solo | SAT_1 | 0.909 / 0.833 / 3 / 0 | 0.875 / 0.714 / 2 / 0 | 0.769 / 0.577 / 3 / 2 |
| Solo | SAT_2 | 0.670 / 0.636 / 26 / 15 | 0.869 / 0.857 / 27 / 3 | 0.871 / 0.862 / 30 / 4 |
| Solo | SAT_3 | 0.774 / 0.686 / 7 / 2 | 0.784 / 0.645 / 4 / 0 | 0.847 / 0.769 / 6 / 0 |
| Solo | SAT_4 | 0.788 / 0.200 / 1 / 4 | 0.651 / 0.111 / 1 / 9 | 0.861 / 0.294 / 1 / 2 |
| Solo | **mean** | 0.785 / 0.589 | 0.795 / 0.582 | 0.837 / 0.625 |
| Plain FL | SAT_1 | 0.661 / 0.476 / 4 / 5 | 0.795 / 0.555 / 2 / 1 | 0.625 / 0.394 / 3 / 5 |
| Plain FL | SAT_2 | 0.216 / 0.191 / 29 / 138 | 0.878 / 0.870 / 31 / 4 | 0.456 / 0.421 / 31 / 49 |
| Plain FL | SAT_3 | 0.713 / 0.633 / 9 / 5 | 0.915 / 0.882 / 9 / 0 | 0.915 / 0.882 / 9 / 0 |
| Plain FL | SAT_4 | 0.746 / 0.270 / 2 / 6 | 0.418 / 0.082 / 2 / 27 | 0.629 / 0.175 / 2 / 11 |
| Plain FL | **mean** | 0.584 / 0.392 | 0.752 / 0.597 | 0.656 / 0.468 |
| FL + personalization | SAT_1 | 0.702 / 0.526 / 4 / 4 | 0.875 / 0.714 / 2 / 0 | 0.892 / 0.832 / 5 / 1 |
| FL + personalization | SAT_2 | 0.864 / 0.852 / 30 / 5 | 0.922 / 0.919 / 31 / 2 | 0.923 / 0.919 / 30 / 2 |
| FL + personalization | SAT_3 | 0.804 / 0.745 / 10 / 3 | 0.818 / 0.714 / 5 / 0 | 0.845 / 0.784 / 8 / 1 |
| FL + personalization | SAT_4 | 0.902 / 0.384 / 1 / 1 | 0.517 / 0.000 / 0 / 15 | 0.948 / 0.555 / 1 / 0 |
| FL + personalization | **mean** | 0.818 / 0.627 | 0.783 / 0.587 | 0.902 / 0.773 |

## Settings chosen

| Arm | Scope | Setting |
|---|---|---|
| Solo | fleet | thr=k5 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Solo | SAT_1 own | thr=a120 ceil=own W=1m spans=s15 close=60 minlen=10 cross=0 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Solo | SAT_2 own | thr=k5 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Solo | SAT_3 own | thr=k2 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=d10 std=0 mask=20 smask=5 latch=0 |
| Solo | SAT_4 own | thr=k5 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Plain FL | fleet | thr=k3 ceil=own W=all spans=s15 close=60 minlen=10 cross=0 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Plain FL | SAT_1 own | thr=a50 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Plain FL | SAT_2 own | thr=k3 ceil=own W=all spans=s15 close=60 minlen=10 cross=0 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| Plain FL | SAT_3 own | thr=k2 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=d10 std=0 mask=20 smask=5 latch=0 |
| Plain FL | SAT_4 own | thr=k1.5 ceil=own W=all spans=s15 close=60 minlen=10 cross=0 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| FL + personalization | fleet | thr=k3 ceil=fleet W=1m spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=10 latch=0 |
| FL + personalization | SAT_1 own | thr=k5 ceil=own W=all spans=s15 close=60 minlen=10 cross=0 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| FL + personalization | SAT_2 own | thr=k3 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
| FL + personalization | SAT_3 own | thr=k2 ceil=own W=all spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=d10 std=0 mask=20 smask=5 latch=0 |
| FL + personalization | SAT_4 own | thr=k5 ceil=own W=1m spans=s15 close=60 minlen=10 cross=0.3 hyst=1.0 drift=0 std=0 mask=20 smask=5 latch=0 |
