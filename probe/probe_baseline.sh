export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=0
export EXP_NAME=cond_mask
export ROOT="/home/junyao/LfHV/NVAE/xm_checkpoints/${EXP_NAME}"
export BETA_STR=100
export BS_STR=32
export CKPT_TIME=03142030
export PORT=6030

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ROOT: ${ROOT}"
echo "EXP_NAME: ${EXP_NAME}"
echo "BETA_STR: ${BETA_STR}"
echo "BS_STR: ${BS_STR}"
echo "CKPT_TIME: ${CKPT_TIME}"
echo "PORT: ${PORT}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python probe_main.py \
--data=/home/junyao/Datasets/xmagical \
--batch_size=64 \
--epochs=100 \
--lr=1e-3 \
--checkpoint=${ROOT}/eval-beta=${BETA_STR}_bs=${BS_STR}_${CKPT_TIME}/checkpoint.pt \
--save=logs/xmagical/${EXP_NAME}/baseline-beta=${BETA_STR}_lr=1e-3_matching \
--master_port=${PORT} \
--matching_loss \
--baseline_mode