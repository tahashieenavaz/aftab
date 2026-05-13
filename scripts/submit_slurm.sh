#!/usr/bin/env bash
set -euo pipefail

mapfile -t ENVS < <(
python - <<'EOF'
from aftab import ENVS
for x in ENVS:
    print(x)
EOF
)

mapfile -t SEEDS < <(
python - <<'EOF'
from aftab import SEEDS
for x in SEEDS:
    print(x)
EOF
)

for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python - <<EOF
from baloot import render_template
render_template(
    "slurm_template",
    "temp",
    ENV="$env",
    SEED="$seed",
)
EOF

    sbatch temp
    rm temp
  done
done