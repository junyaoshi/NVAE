export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=0
export ROOT=/home/junyao/LfHV/NVAE/xmagical_checkpoints/hierarchical-lite
export EXP_NAME=untrained_baseline
export PROBE_TIME=20220216_0200
export CKPT_NUM=000
export PORT=6020

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
--save=logs/baseline/hierarchical-lite_probe-ckpt=${CKPT_NUM}_lr=1e-3_matching \
--master_port=${PORT} \
--matching_loss \
--baseline_mode