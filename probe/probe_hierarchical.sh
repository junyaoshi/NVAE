export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=1
export ROOT=/home/junyao/LfHV/NVAE/xmagical_checkpoints/hierarchical-lite
export EXP_NAME=hierarchical_lite_mask_process
export BETA_STR=20
export CKPT_TIME=02141930
export CKPT_NUM=146
export PORT=6021

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ROOT: ${ROOT}"
echo "EXP_NAME: ${EXP_NAME}"
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
--checkpoint=${ROOT}/eval-${EXP_NAME}_beta=${BETA_STR}_${CKPT_TIME}/checkpoint_${CKPT_NUM}.pt \
--save=logs/${EXP_NAME}_probe/probe-${EXP_NAME}_beta=${BETA_STR}_ckpt=${CKPT_NUM}_lr=1e-3_matching \
--master_port=${PORT} \
--matching_loss