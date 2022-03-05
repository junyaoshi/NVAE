#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-2080ti-5
#SBATCH --job-name=xm-0
#SBATCH -o out/xm_zero_latent.out

export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=0
export BETA_STR=0
export SEED=20
export BATCH_SIZE=16
export MASTER_PORT=6020
export EPOCHS=300
export NUM_WORKERS=4

echo "PYTHONPATH: ${PYTHONPATH}"
echo "BETA: ${BETA}"
echo "BETA_STR: ${BETA_STR}"
echo "SEED: ${SEED}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "EPOCHS: ${EPOCHS}"
echo "NUM_WORKERS: ${NUM_WORKERS}"

python /home/junyao/LfHV/NVAE/train_dair.py \
--root=/scratch/junyao/LfHV/NVAE/xm_checkpoints/cond_mask/zero_latent \
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
--save=beta=${BETA_STR}_bs=${BATCH_SIZE}_03012300 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_robot_mask --process_cond_info \
--num_workers=${NUM_WORKERS} \
--zero_latent \

wait