export CUDA_VISIBLE_DEVICES=1
export BETA=0.03
export GAMMA=100.0
export TYPE=conditional_adversary

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "BETA: ${BETA}"
echo "GAMMA: ${GAMMA}"
echo "TYPE: ${TYPE}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python main.py \
--save=logs/${TYPE}/beta=${BETA}_gamma=${GAMMA} \
--batch_size=128 \
--vae_epochs=20 \
--probe_epochs=100 \
--vae_lr=1e-3 \
--probe_lr=1e-3 \
--num_features=2 \
--num_samples=5 \
--latent_dim=2 \
--conditional \
--cond_dim=1 \
--beta=${BETA} \
--gamma=${GAMMA} \
--adversary
