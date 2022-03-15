#!/bin/bash
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH -w node-3090-0
#SBATCH --job-name=ss-0
#SBATCH -o out/ss_hand_0.out

export PYTHONPATH="/home/junyao/LfHV/NVAE"
export BETA=0
export BETA_STR=0
export SEED=42
export BATCH_SIZE=32
export MASTER_PORT=6022
export EPOCHS=300
export NUM_WORKERS=4
export TYPE="cond_hand/zero_latent"
export TIME="03071800"

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
--root=/scratch/junyao/LfHV/NVAE/ss_checkpoints/${TYPE} \
--dataset=something-something \
--data=/scratch/agelosk/Hands/something_something_paths.pkl \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--weight_decay_norm=1e-1 \
--num_nf=1 \
--num_latent_scales=2 --num_groups_per_scale=12 --num_latent_per_group=20 \
--num_channels_enc=64 --num_channels_dec=64 \
--num_preprocess_blocks=2 --num_preprocess_cells=2 \
--num_postprocess_blocks=2 --num_postprocess_cells=8 \
--num_cell_per_cond_enc=1 --num_cell_per_cond_dec=1 \
--num_process_per_node=1 \
--use_se --res_dist --ada_groups \
--num_workers=${NUM_WORKERS} \
--save=beta=${BETA_STR}_bs=${BATCH_SIZE}_${TIME} \
--master_port=${MASTER_PORT} \
--seed=${SEED} \
--kl_beta=${BETA} \
--cond_hand --process_cond_info \
--num_workers=${NUM_WORKERS} \
--zero_latent \

wait