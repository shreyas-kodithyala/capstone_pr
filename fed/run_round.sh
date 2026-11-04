#!/bin/zsh
# One full round after the dataset exists: validation, baselines, solo/fed/pers + detection, RUL.
# Usage: caffeinate -i zsh fed/run_round.sh   (from New/)
set -e
cd "$(dirname "$0")/.."
mkdir -p fed/out_v2 data/sim_fleet/validation
(cd sim && nice -n 15 conda run -n bsk python validate.py > ../data/sim_fleet/validation/validate.out 2>&1) &
(cd sim && nice -n 15 conda run -n timeeval python baselines.py > ../data/sim_fleet/validation/baselines.out 2>&1) &
(cd fed && nice -n 15 conda run -n timeeval python rul.py > out_v2/rul.out 2>&1) &
(cd fed && nice -n 10 conda run -n timeeval --no-capture-output python fedsim.py > out_v2/run.out 2>&1)
wait
echo "round done"
