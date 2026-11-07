#!/usr/bin/env bash
# 武侠风格微调流水线：DAPT（继续预训练）→ SFT（对话/续写）→ 盲评材料
#
# 前置：
#   * 已有 d6 基座：$NANOCHAT_BASE_DIR/base_checkpoints/d6-zh（step 300）
#   * 已解压云盘数据：
#       /mnt/workspace/nanochat-data/data_wuxia_jinyong
#       /mnt/workspace/nanochat-data/wuxia_sft
#
# 用法：
#   bash runs/wuxia_finetune.sh
#   WUXIA_DATA_DIR=/path/to/wuxia SFT_DATA_DIR=/path/to/sft bash runs/wuxia_finetune.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
WUXIA_DATA="${WUXIA_DATA_DIR:-/mnt/workspace/nanochat-data/data_wuxia_jinyong}"
SFT_DATA="${SFT_DATA_DIR:-/mnt/workspace/nanochat-data/wuxia_sft}"
BASE_TAG="${BASE_TAG:-d6-zh}"
BASE_STEP="${BASE_STEP:-300}"
DAPT_STEPS="${DAPT_STEPS:-500}"
SFT_STEPS="${SFT_STEPS:-2500}"

export NANOCHAT_BASE_DIR="$BASE_DIR"
export NANOCHAT_DISABLE_VE=1
export NANOCHAT_DISABLE_COMPILE=1
export NANOCHAT_ENFORCE_EAGER=1
export WANDB_MODE=offline

if [[ ! -f "$BASE_DIR/base_checkpoints/$BASE_TAG/model_$(printf '%06d' "$BASE_STEP").pt" ]]; then
    echo "找不到基座 checkpoint：$BASE_DIR/base_checkpoints/$BASE_TAG step $BASE_STEP" >&2
    exit 1
fi

# 1) DAPT：复制基座 checkpoint，换 tag 后从 step 300 继续训练
if [[ ! -d "$BASE_DIR/base_checkpoints/wuxia-dapt" ]]; then
    echo "[1/3] 复制基座 checkpoint -> wuxia-dapt"
    cp -r "$BASE_DIR/base_checkpoints/$BASE_TAG" "$BASE_DIR/base_checkpoints/wuxia-dapt"
fi

echo "[1/3] DAPT：wuxia 语料，500 steps（低学习率）"
NANOCHAT_DATA_DIR="$WUXIA_DATA" python3 -m scripts.base_train \
    --depth=6 --max-seq-len=512 --device-batch-size=16 --total-batch-size=32768 \
    --num-iterations="$DAPT_STEPS" --resume-from-step="$BASE_STEP" \
    --matrix-lr=0.005 --embedding-lr=0.05 --unembedding-lr=0.002 --scalar-lr=0.1 \
    --eval-every=100 --eval-tokens=1000000 --sample-every=-1 --save-every=100 \
    --model-tag=wuxia-dapt --run=wuxia-dapt

# 2) SFT：assistant-only loss，保存到 chatsft_checkpoints/wuxia-sft
echo
echo "[2/3] SFT：$SFT_STEPS steps"
python3 scripts/sft_train.py \
    --base-tag wuxia-dapt --base-step "$DAPT_STEPS" \
    --data-dir "$SFT_DATA" --out-tag wuxia-sft \
    --iters "$SFT_STEPS" --val-every 500 --batch 8 --seq 512

# 3) 盲评材料
echo
echo "[3/3] 生成盲评材料"
python3 scripts/eval_wuxia.py --n 3 --max-tokens 120

cat <<EOF

==============================================================
 完成。推理：
   python -m scripts.chat_cli -i sft -g wuxia-sft -p "师父，陆老前辈，咱们拿这恶贼怎么办？"
 盲评表：
   \$NANOCHAT_BASE_DIR/eval/wuxia_blind_eval.md
 答案 key（先别打开）：
   \$NANOCHAT_BASE_DIR/eval/wuxia_blind_key.json
==============================================================
EOF
