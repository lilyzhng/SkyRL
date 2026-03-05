#!/bin/bash
# Step 3 (test): 小规模训练测试
# 100 tasks, 1 epoch, Qwen3-8B, GRPO
# 预计 ~30-45 min on 4x H100

set -ex

cd "$(dirname "$0")"

MODAL_GPU=H100:4 MODAL_TIMEOUT=14400 modal run --detach examples/train_integrations/modal/main.py \
  --command "
    # 1. 准备完整数据集
    uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests

    # 2. 创建 100-task 子集
    FULL_DIR=\$HOME/data/harbor/CodeContests
    SUBSET_DIR=\$HOME/data/harbor/CodeContests-100
    mkdir -p \$SUBSET_DIR
    for d in \$(ls \$FULL_DIR | head -100); do
      ln -s \$FULL_DIR/\$d \$SUBSET_DIR/\$d
    done
    echo \"Created 100-task subset at \$SUBSET_DIR\"

    # 3. 开始训练
    DAYTONA_API_KEY=dtn_41c7109e106fd349b61ee8c9258e427c69a5e514143743b2399c1d0e00909db8 \
    WANDB_API_KEY=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 \
    bash examples/train_integrations/harbor/run_codecontest.sh \
      data.train_data=\"['\$HOME/data/harbor/CodeContests-100']\" \
      data.val_data=null \
      trainer.eval_interval=0 \
      trainer.eval_before_train=false \
      trainer.epochs=1
  "
