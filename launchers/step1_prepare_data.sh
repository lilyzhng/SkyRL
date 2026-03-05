#!/bin/bash
# Step 1: 准备 dataset
# 从 HuggingFace 下载 CodeContests，解压到 Modal persistent volume
# 必须写到 /root/data/（volume 挂载点），不然下次容器启动数据就丢了
# 路径 /root/data/harbor/CodeContests 匹配 run_harbor_gen.sh 的期望

set -ex

cd "$(dirname "$0")"

modal run examples/train_integrations/modal/main.py \
  --command "uv run examples/train_integrations/harbor/prepare_harbor_dataset.py --dataset open-thoughts/CodeContests --output_dir /root/data/harbor/CodeContests"
