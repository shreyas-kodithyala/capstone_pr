#!/bin/bash
# Final frozen run on this machine: waits for the fresh fleet, then validates, scores once, reports.
cd "$(dirname "$0")/.." && source .localenv/activate.sh
unset SIM_FLEET FED_FIT_DAYS
export FED_OUT=$CAP/fed/out_final
TE=$L/miniforge/envs/timeeval/bin/python; BK=$L/miniforge/envs/bsk/bin/python
until grep -q "^end " data/gen_final.log; do sleep 15; done
grep -q "rc=0" data/gen_final.log || { echo "generation failed"; exit 1; }
mkdir -p $FED_OUT/reports data/sim_fleet/validation
(cd sim && nice -n 15 $BK validate.py > ../data/sim_fleet/validation/validate.out 2>&1) &
(cd sim && nice -n 15 $TE baselines.py > ../data/sim_fleet/validation/baselines.out 2>&1) &
(cd sim && nice -n 15 $TE dataset_stats.py > ../data/sim_fleet/validation/dataset_stats.txt 2>&1) &
(cd fed && nice -n 15 $TE rul.py > $FED_OUT/rul.out 2>&1) &
cd fed
echo "start $(date +%H:%M:%S)" > $FED_OUT/run.out
caffeinate -i nice -n 10 $TE fedsim.py >> $FED_OUT/run.out 2>&1; rc=$?
echo "end $(date +%H:%M:%S) rc=$rc" >> $FED_OUT/run.out
wait
[ $rc = 0 ] || exit $rc
nice -n 5 $TE causal_check.py fed federated $FED_OUT > $FED_OUT/causal_fed.out 2>&1
nice -n 5 $TE causal_check.py solo local $FED_OUT > $FED_OUT/causal_solo.out 2>&1
$TE report_round.py $FED_OUT > $FED_OUT/reports/final.txt 2>&1
echo "final report written"
