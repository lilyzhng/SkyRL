#!/bin/bash
# Quick test: 10 train tasks + 5 eval tasks
# Purpose: verify HTML trajectory logging works on WandB
# Should complete in ~15-20 min

set -ex

cd "$(dirname "$0")"

MODAL_GPU=H100:4 MODAL_TIMEOUT=7200 modal run --detach examples/train_integrations/modal/main.py \
  --command "\
    find /root/data/harbor/CodeContests/ -maxdepth 2 -type l -delete 2>/dev/null; \
    echo '=== Creating 5-task eval set ===' && \
    rm -rf /root/data/harbor/CodeContests-eval && \
    mkdir -p /root/data/harbor/CodeContests-eval && \
    ls /root/data/harbor/CodeContests/ | sort | head -5 | while read d; do \
      cp -r /root/data/harbor/CodeContests/\$d /root/data/harbor/CodeContests-eval/\$d; \
    done && \
    echo '=== Creating 10-task train set ===' && \
    rm -rf /root/data/harbor/CodeContests-10 && \
    mkdir -p /root/data/harbor/CodeContests-10 && \
    ls /root/data/harbor/CodeContests/ | sort | tail -n +6 | head -10 | while read d; do \
      cp -r /root/data/harbor/CodeContests/\$d /root/data/harbor/CodeContests-10/\$d; \
    done && \
    echo \"Eval: \$(ls /root/data/harbor/CodeContests-eval/ | wc -l), Train: \$(ls /root/data/harbor/CodeContests-10/ | wc -l)\" && \
    SKYRL_DATA_DIR=/root/data/harbor \
    SKYRL_RUN_NAME=\"quick-test-html-\$(date +%m%d-%H%M)\" \
    bash examples/train_integrations/harbor/run_codecontest.sh \
      data.train_data=\"['/root/data/harbor/CodeContests-10']\" \
      data.val_data=\"['/root/data/harbor/CodeContests-eval']\" \
      trainer.eval_before_train=true \
      trainer.eval_interval=1 \
      trainer.epochs=1 \
      trainer.train_batch_size=10 \
      trainer.policy_mini_batch_size=10 \
  "
