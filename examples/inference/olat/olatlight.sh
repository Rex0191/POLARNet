#!/bin/bash
# infer_relit.sh
# 批量或单张运行 relighting 推理
# Author: ChatGPT (SJTU version)

# ===== 环境配置 =====
export HF_HOME=/mnt/bn/pico-idl-avatar2/cz/LBM/hugging_face
export HF_ENDPOINT=https://hf-mirror.com

# ===== 输入输出路径 =====
INPUT_DIR="/mnt/bn/idl-data-cache/cz/LBM/examples/inference/ckpts/relighting/assets/woman_4"
MASK_DIR="/mnt/bn/idl-data-cache/cz/LBM/examples/inference/ckpts/relighting/assets/mask_resized"
OUTPUT_DIR="/mnt/bn/idl-data-cache/cz/LBM/examples/inference/ckpts/olat/assets/show_woman_4_test"

# ===== 推理参数 =====
MODEL_NAME="olat"
NUM_STEPS=1
DEVICE="cuda"
BF16="--bf16"  # 若要关闭 BF16 用 float16，改为：BF16=""

# ===== 创建输出目录 =====
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting batch olatlighting inference..."
echo "Input dir:  $INPUT_DIR"
echo "Mask dir:   $MASK_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "--------------------------------------------"

# ===== 执行推理 =====
CUDA_VISIBLE_DEVICES=1 python olatlight.py \
  --input_dir "$INPUT_DIR" \
  --mask_dir "$MASK_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --num_inference_steps $NUM_STEPS \
  --device $DEVICE \
  $BF16

echo "✅ 全部推理完成！结果已保存至: $OUTPUT_DIR"
