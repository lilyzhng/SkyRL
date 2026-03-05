#!/bin/bash
# Step 3: 完整训练
# Qwen3-8B, GRPO, 3 epochs, CodeContests dataset
# 需要 4x H100 GPU，跑很久，记得用 --detach
#
# Prerequisites:
#   - Modal secrets: wandb-secret, daytona-secret (configured in main.py)
#   - CodeContests dataset on skyrl-data volume (already prepared)
#   - OpenThoughts-TB-dev eval dataset will be prepared automatically if missing

set -ex

cd "$(dirname "$0")"

# Prepare eval dataset if missing, then run training
MODAL_GPU=H100:4 MODAL_TIMEOUT=14400 modal run --detach examples/train_integrations/modal/main.py \
  --command "SKYRL_DATA_DIR=/root/data/harbor SKYRL_RUN_NAME=cc-full-3roll-\$(date +%m%d-%H%M) bash examples/train_integrations/harbor/run_codecontest.sh"
