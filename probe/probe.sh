export PYTHONPATH=/home/junyao/LfHV/NVAE:/home/junyao/LfHV/NVAE/probe
echo $PYTHONPATH
CUDA_VISIBLE_DEVICES=2 \
python probe_main.py \
--data=/home/junyao/Datasets/xirl/xmagical \
--batch_size=64 \
--epochs=50 \
--lr=1e-3 \
--checkpoint=/home/junyao/LfHV/NVAE/checkpoints/eval-vanilla_mask_process_beta=4_01260010/checkpoint_013.pt \
--save=logs/probe-vanilla_mask_process_beta=4_01260010_lr=1e-3_all \
--master_port=6022 \
--matching_loss
