export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=0
export ROOT=/home/junyao/LfHV/NVAE/xmagical_checkpoints
export EXP_NAME=debug
export PROBE_TIME=20220215_1700
export CKPT_NUM=000
export PORT=6025

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ROOT: ${ROOT}"
echo "EXP_NAME: ${EXP_NAME}"
echo "PROBE_TIME: ${PROBE_TIME}"
echo "CKPT_NUM: ${CKPT_NUM}"
echo "PORT: ${PORT}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python probe_main.py \
--data=/home/junyao/Datasets/xirl/xmagical \
--batch_size=256 \
--epochs=100 \
--lr=1e-3 \
--checkpoint=${ROOT}/eval-${EXP_NAME}/checkpoint_${CKPT_NUM}.pt \
--save=logs/probe_debug/${PROBE_TIME}_probe_debug/probe-${EXP_NAME}_ckpt=${CKPT_NUM}_lr=1e-3 \
--master_port=${PORT}