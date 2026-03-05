#!/bin/bash
# Step 2: Generation-only 测试
# 用小模型 Qwen2.5-1.5B-Instruct 跑 10 个 sample，确认 pipeline 通了再上完整训练
# 需要 4x H100 GPU
# 在同一个容器里先准备数据再跑，避免跨容器持久化问题

set -ex

cd "$(dirname "$0")"

MODAL_GPU=H100:4 modal run --detach examples/train_integrations/modal/main.py \
  --command "uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests && DAYTONA_API_KEY=dtn_41c7109e106fd349b61ee8c9258e427c69a5e514143743b2399c1d0e00909db8 WANDB_API_KEY=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 bash examples/train_integrations/harbor/run_harbor_gen.sh data.val_data=null trainer.eval_interval=0"
