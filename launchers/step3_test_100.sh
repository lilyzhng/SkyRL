#!/bin/bash
# Step 3 (test): 100-sample GRPO training test with 20-task eval
# Qwen3-8B, GRPO, 3 epochs, 100 CodeContests tasks, 3 rollouts per task
# Fixed 20-task eval set (non-overlapping with train) for tracking progress on WandB
# 4x H100 GPU, estimated ~1.5-2 hours
#
# Data split (sorted by name, deterministic):
#   - First 20 tasks  → eval  (CodeContests-eval)
#   - Next 100 tasks  → train (CodeContests-100)

set -ex

cd "$(dirname "$0")/.."

MODAL_GPU=H100:4 MODAL_TIMEOUT=14400 modal run --detach examples/train_integrations/modal/main.py \
  --command "\
    echo '=== Cleaning up corrupted symlinks ===' && \
    find /root/data/harbor/CodeContests/ -maxdepth 2 -type l -delete 2>/dev/null; \
    echo '=== Creating fixed eval set (first 20 tasks) ===' && \
    rm -rf /root/data/harbor/CodeContests-eval && \
    mkdir -p /root/data/harbor/CodeContests-eval && \
    ls /root/data/harbor/CodeContests/ | sort | head -20 | while read d; do \
      cp -r /root/data/harbor/CodeContests/\$d /root/data/harbor/CodeContests-eval/\$d; \
    done && \
    echo \"Eval size: \$(ls /root/data/harbor/CodeContests-eval/ | wc -l) tasks\" && \
    echo '=== Creating fixed train set (next 100 tasks) ===' && \
    rm -rf /root/data/harbor/CodeContests-100 && \
    mkdir -p /root/data/harbor/CodeContests-100 && \
    ls /root/data/harbor/CodeContests/ | sort | tail -n +21 | head -100 | while read d; do \
      cp -r /root/data/harbor/CodeContests/\$d /root/data/harbor/CodeContests-100/\$d; \
    done && \
    echo \"Train size: \$(ls /root/data/harbor/CodeContests-100/ | wc -l) tasks\" && \
    echo '=== Starting training ===' && \
    SKYRL_DATA_DIR=/root/data/harbor \
    SKYRL_RUN_NAME=\"cc-100t-20eval-3roll-\$(date +%m%d-%H%M)\" \
    bash examples/train_integrations/harbor/run_codecontest.sh \
      data.train_data=\"['/root/data/harbor/CodeContests-100']\" \
      data.val_data=\"['/root/data/harbor/CodeContests-eval']\" \
      trainer.eval_before_train=true \
      trainer.eval_interval=5 \
      trainer.train_batch_size=10 \
      trainer.policy_mini_batch_size=10 \
      generator.inference_engine.gpu_memory_utilization=0.4 \
  "
