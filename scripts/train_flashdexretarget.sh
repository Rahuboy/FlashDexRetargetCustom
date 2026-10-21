#!/bin/bash
# MOTION=<motion.pt> [ROBOT=xhand|sharpa] [NPROC=<gpus>] bash scripts/train_flashdexretarget.sh [hydra overrides ...]
# NPROC > 1: data-parallel over that many GPUs (torchrun; src/common/distributed.py): num_envs envs per GPU, rank 0
# logs, evaluates and checkpoints, the other ranks' output goes to logs/torchrun/.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
unset HEADLESS   # Isaac Lab's own env var; use the headless=false override instead

robot="${ROBOT:-xhand}"
motion="$(realpath "${MOTION:?set MOTION=<dataset>/motion.pt}")"

dir="$(dirname "$motion")"
[ "$(basename "$dir")" = "$robot" ] && dir="$(dirname "$dir")"
dataset="$(basename "$dir")"

args=(--overrides envs/robot@robot=${robot}_bimanual)
if [ "$robot" = sharpa ]; then
    unset FDR_ROBOT_USD   # env.sh points it at the XHAND USD
fi
args+=(
    --overrides commands.motion_file="$motion"
    --overrides logger.wandb.run_name=flashdexretarget-${dataset}-${robot}
    --overrides "logger.wandb.tags=[${dataset},${robot}]"
)
for ov in "$@"; do args+=(--overrides "$ov"); done

if [ "${NPROC:-1}" -gt 1 ]; then
    exec torchrun --standalone --nproc_per_node="$NPROC" --log-dir logs/torchrun \
        --redirects "$(seq -s, -f '%g:3' 1 $((NPROC - 1)))" train.py "${args[@]}"
fi
python train.py "${args[@]}"
