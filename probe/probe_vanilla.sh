export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=0
export ROOT=/home/junyao/LfHV/NVAE/xmagical_checkpoints/vanilla-lite
export EXP_NAME=vanilla_lite_mask_process_beta
export PROBE_TIME=20220214_1830
export BETA_STR=5
export CKPT_TIME=02140200
export CKPT_NUM=000
export PORT=6020

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ROOT: ${ROOT}"
echo "EXP_NAME: ${EXP_NAME}"
echo "PROBE_TIME: ${PROBE_TIME}"
echo "BETA_STR: ${BETA_STR}"
echo "CKPT_NUM: ${CKPT_NUM}"
echo "CKPT_TIME: ${CKPT_TIME}"
echo "PORT: ${PORT}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python probe_main.py \
--data=/home/junyao/Datasets/xirl/xmagical \
--batch_size=256 \
--epochs=100 \
--lr=1e-3 \
--checkpoint=${ROOT}/eval-${EXP_NAME}=${BETA_STR}_${CKPT_TIME}/checkpoint_${CKPT_NUM}.pt \
--save=logs/${PROBE_TIME}_probe_debug/probe-${EXP_NAME}=${BETA_STR}_${CKPT_TIME}_lr=1e-3 \
--master_port=${PORT} \
--matching_loss
