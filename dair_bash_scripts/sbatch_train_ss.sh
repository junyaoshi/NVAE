#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=ss-b10
#SBATCH -o out/ss_cond_hand_b10.out

export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=10
export BETA_STR=10
export SEED=17
export BATCH_SIZE=16
export MASTER_PORT=6012
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
--root=/scratch/junyao/LfHV/NVAE/ss_checkpoints/cond_hand/zero_latent \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something_paths.pkl \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=12 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=3 --num_preprocess_cells=2 \
--num_postprocess_blocks=3 --num_postprocess_cells=20 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --ada_groups \
--num_workers=${NUM_WORKERS} \
--save=beta=${BETA_STR}_bs=${BATCH_SIZE}_03012300 \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_hand --process_cond_info \
--num_workers=${NUM_WORKERS}

wait