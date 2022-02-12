export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
export CUDA_VISIBLE_DEVICES=0
export BETA_STR=4
export CKPT_NUM=087
export PORT=6020

echo "PYTHONPATH: ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "BETA_STR: ${BETA_STR}"
echo "CKPT_NUM: ${CKPT_NUM}"
echo "PORT: ${PORT}"


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python probe_main.py \
--data=/home/junyao/Datasets/xirl/xmagical \
--batch_size=256 \
--epochs=100 \
--lr=1e-3 \
--checkpoint=/home/junyao/LfHV/NVAE/checkpoints/mask/eval-vanilla_mask_process_beta=${BETA_STR}_01260010/checkpoint_${CKPT_NUM}.pt \
--save=logs/20220211_probe/probe-vanilla_mask_process_beta=${BETA_STR}_01260010_lr=1e-3 \
--master_port=${PORT} \
--matching_loss
