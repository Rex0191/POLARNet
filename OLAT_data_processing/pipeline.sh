#!/bin/bash
set -e
set -o pipefail

export HF_HOME=/mnt/bn/cz/Matte-Anything/hugging_face # Your hugging face path
export HF_ENDPOINT=https://hf-mirror.com


# source /mnt/bn/pico-idl-avatar/cz/miniconda3/etc/profile.d/conda.sh

# ========= 全局路径配置 =========
# ROOT_IN="/mnt/hdfs/haruna/sjtu"
ROOT_OUT="POLAR/ori_OLAT"
ROOT_SEG=$(echo "${ROOT_OUT}" | sed 's/ori_OLAT/for_segmentation/')
ROOT_ALPHA=$(echo "${ROOT_SEG}" | sed 's/for_segmentation/alpha_data/')
ROOT_UNIFORM=$(echo "${ROOT_OUT}" | sed 's/ori_OLAT/uniform/')
IDS_FILE="ids.txt"

# ========= Step 2 路径配置 =========
OLAT_TXT="OLAT_data_processing/light/light_157_proc.txt"
BASE_MAP_DIR="OLAT_data_processing/light/OLAT_EnvMaps/"
ENVMAP_DIR="OLAT_data_processing/light/hdrs_all/"
BACKGROUND_DIR="OLAT_data_processing/light/hdr_background/"
UNI_ENVMAP_DIR="OLAT_data_processing/light/hdrs_uniform/"

# ========= GPU 配置 =========
GPUS="1,2,3,4,5,6,7,8"

# ========= 步骤开关 =========
SKIP_STEP1=true
SKIP_STEP15=false
SKIP_STEP2=false
SKIP_STEP3=false
SKIP_STEP4=false

# ========= Step 1: 数据预处理 =========
task_preprocess() {
    if [ "$SKIP_STEP1" = true ]; then
        echo "[STEP 1] 跳过数据预处理."
        return
    fi
    echo "[STEP 1] 数据预处理开始..."
    conda activate delit
    parallel -j 64 --eta --results logs_parallel --joblog joblog.txt \
      "CUDA_VISIBLE_DEVICES=${GPUS} python preprocess.py \
         -i ${ROOT_IN}/{1} \
         -o ${ROOT_OUT}/{1} \
         -r --keep-structure \
         --gamma 1.0 --contrast linear --alpha 1.0 --beta 0 --overwrite" \
      :::: "${IDS_FILE}" | tee logs_step1.txt
    echo "[STEP 1] 数据预处理完成."
}

# ========= Step 1.5: 路径整理 (移除 Photos 文件夹) =========
task_fix_structure() {
    if [ "$SKIP_STEP15" = true ]; then
        echo "[STEP 1.5] 跳过目录整理."
        return
    fi
    echo "[STEP 1.5] 开始整理目录结构..."

    pushd "${ROOT_OUT}" > /dev/null
    find . -type d -name "Photos" | while read p; do
        parent=$(dirname "$p")
        echo "[INFO] 移动 $p/* 到 $parent/"
        mv "$p"/* "$parent"/
        rmdir "$p"
    done
    popd > /dev/null   # ← 回到原来的脚本目录

    echo "[STEP 1.5] 目录整理完成."
}


# ========= Step 2: 均匀光合成（带背景） =========
task_uniform_with_bg() {
    if [ "$SKIP_STEP2" = true ]; then
        echo "[STEP 2] 跳过均匀光合成."
        return
    fi
    echo "[STEP 2] 均匀光合成开始..."
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    SEGMENT_DIR="${SCRIPT_DIR}/segment"

    CUDA_VISIBLE_DEVICES=${GPUS} python "${SEGMENT_DIR}/batch_uniform_with_bg.py" \
        --root_dir "${ROOT_OUT}" \
        --root_out "${ROOT_UNIFORM}" \
        --olat_txt "${OLAT_TXT}" \
        --base_map_dir "${BASE_MAP_DIR}" \
        --envmap_dir "${UNI_ENVMAP_DIR}" \
        --background_dir "${BACKGROUND_DIR}" \
        --num_envmaps 1 \
        --num_workers 16 | tee logs_step2.txt
    echo "[STEP 2] 均匀光合成完成."
}

# ========= Step 3: 抠图 (Matting) =========
task_matting() {
    if [ "$SKIP_STEP3" = true ]; then
        echo "[STEP 3] 跳过人像抠图."
        return
    fi
    echo "[STEP 3] 人像抠图开始..."
    conda activate lbm

    pushd /mnt/bn/idl-data-cache/cz/Matte-Anything > /dev/null
    CUDA_VISIBLE_DEVICES=${GPUS} python batch_matte.py \
        -i "${ROOT_SEG}" \
        -o "${ROOT_ALPHA}" \
        -r --keep_structure \
        --fg-caption "person" \
        --gpu-ids ${GPUS} | tee logs_step3_matting.txt
    popd > /dev/null   # 回到 main_pipeline.sh 所在目录

    echo "[STEP 3] 人像抠图完成."
}


# ========= Step 4: 彩色光合成 =========
task_color_light() {
    if [ "$SKIP_STEP4" = true ]; then
        echo "[STEP 4] 跳过彩色光合成."
        return
    fi
    echo "[STEP 4] 彩色光合成开始..."
    conda activate delit
    SCRIPT_DIR="$(dirname "$(realpath "$0")")"
    COLOR_DIR="${SCRIPT_DIR}/color"

    ROOT_SYNTH=$(echo "${ROOT_OUT}" | sed 's/ori_OLAT/synthetic_image/')

    CUDA_VISIBLE_DEVICES=${GPUS} python "${COLOR_DIR}/batch_synthesis_color_light.py" \
        --root_dir "${ROOT_OUT}" \
        --root_alpha "${ROOT_ALPHA}" \
        --root_out "${ROOT_SYNTH}" \
        --olat_txt "${OLAT_TXT}" \
        --base_map_dir "${BASE_MAP_DIR}" \
        --envmap_dir "${ENVMAP_DIR}" \
        --background_dir "${BACKGROUND_DIR}" \
        --num_envmaps 800 \
        --num_workers 16 | tee logs_step4_colorlight.txt
    echo "[STEP 4] 彩色光合成完成."
}

# ========= 主流程 =========
main() {
    task_preprocess
    task_fix_structure
    task_uniform_with_bg
    task_matting
    task_color_light
}

main "$@"
