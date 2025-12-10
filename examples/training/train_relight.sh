export HF_HOME=/mnt/bn/pico-idl-avatar2/cz/LBM/hugging_face
export HF_ENDPOINT=https://hf-mirror.com

export SLURM_JOB_ID=local
export SLURM_ARRAY_TASK_ID=0
export SLURM_PROCID=0
export SLURM_NPROCS=1
export SLURM_NNODES=1

CUDA_VISIBLE_DEVICES=1 python train_lbm_relight.py --path_config config/relight_olat.yaml
