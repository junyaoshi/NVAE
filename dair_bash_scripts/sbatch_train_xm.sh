#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=xm-b1
#SBATCH -o out/xm_mask_b1.out

export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=1
export BETA_STR=1
export SEED=25
export BATCH_SIZE=16
export MASTER_PORT=6025
export EPOCHS=300
export NUM_WORKERS=4
export TYPE="cond_mask"
export TIME="03020930"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "BETA: ${BETA}"
echo "BETA_STR: ${BETA_STR}"
echo "SEED: ${SEED}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "TYPE: ${TYPE}"
echo "TIME: ${TIME}"

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/LfHV/NVAE/xm_checkpoints/${TYPE} \
--dataset=xmagical \
--data=/scratch/junyao/Datasets/xmagical \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=1 --num_groups_per_scale=10 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=3 --num_preprocess_cells=2 \
--num_postprocess_blocks=3 --num_postprocess_cells=12 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --ada_groups \
--num_workers=${NUM_WORKERS} \
--save=beta=${BETA_STR}_bs=${BATCH_SIZE}_${TIME} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--num_workers=${NUM_WORKERS} \
--cond_robot_mask --process_cond_info \

wait