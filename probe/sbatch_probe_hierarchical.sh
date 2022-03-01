#!/bin/bash
#SBATCH --mem-per-gpu=12G
#SBATCH --partition=dineshj-compute
#SBATCH --qos=dineshj-high
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --gpus=rtx2080ti:1
#SBATCH --job-name=pr-vm
#SBATCH -w node-2080ti-5
#SBATCH -o out/probe_xm_vm_h-med_beta10.out

export PYTHONPATH="/home/junyao/LfHV/NVAE"
export EXP_NAME="vec_mask"
ROOT=/scratch/junyao/LfHV/NVAE/xmagical_checkpoints/hierarchical-med/${EXP_NAME}
BETA_STR=10
CKPT_TIME=02212100
CKPT_NUM=034
PORT=6032
NUM_WORKERS=4

echo "PYTHONPATH: ${PYTHONPATH}"

python /home/junyao/LfHV/NVAE/probe/probe_main.py \
--data=/scratch/junyao/Datasets/xmagical \
--batch_size=50 \
--epochs=100 \
--lr=1e-3 \
--num_workers=${NUM_WORKERS} \
--checkpoint=${ROOT}/eval-beta=${BETA_STR}_bs=25_${CKPT_TIME}/checkpoint_${CKPT_NUM}.pt \
--save=/scratch/junyao/LfHV/NVAE/probe/xmagical/hierarchical-med/${EXP_NAME}_beta=${BETA_STR}_ckpt=${CKPT_NUM}_matching \
--master_port=${PORT} \
--matching_loss

wait