import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from model import LinearVAE, LinearProbe
from datasets import ToyRandomDataset

"""
Framework:
x = (x1, x2) -> z -> concat z with c = x2 -> x' = (x1', x2')

We want to preserve x1 while eliminating x2 from the latent z
"""

parser = argparse.ArgumentParser(description='DAIR MLP')
parser.add_argument('--save', type=str, default="logs",
                    help='directory for saving run information')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--vae_epochs', type=int, default=10,
                    help='number of epochs to train vae (default: 10)')
parser.add_argument('--probe_epochs', type=int, default=20,
                    help='number of epochs to train probe (default: 20)')
parser.add_argument('--vae_lr', type=float, default=1e-3,
                    help='learning rate for training vae (default: 1e-3)')
parser.add_argument('--probe_lr', type=float, default=1e-3,
                    help='learning rate for training probe (default: 1e-3)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before printing training status')
parser.add_argument('--num_features', type=int, default=2,
                    help='number of features')
parser.add_argument('--num_samples', type=int, default=8,
                    help='number of samples to generate during testing')
parser.add_argument('--latent_dim', type=int, default=1,
                    help='dimension of VAE latent')
parser.add_argument('--conditional', action='store_true', default=False,
                    help='enables conditional VAE')
parser.add_argument('--cond_dim', type=int, default=1,
                    help='dimension of conditional input (when VAE is conditional)')
parser.add_argument('--adversary', action='store_true', default=False,
                    help='enables adversarial discriminator loss')
parser.add_argument('--beta', type=float, default=1.0,
                    help='beta value of BetaVAE')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='gamma coefficient for adversarial loss')
parser.add_argument('--debug', action='store_true', default=False,
                    help='enables debugging mode, set num_workers=0 to allow for break points')

args = parser.parse_args()
assert args.num_samples <= args.batch_size
assert args.cond_dim < args.num_features, "conditional input dimension must be less than feature dimension"
if args.adversary:
    assert args.conditional, "adversary must be used on conditional vae"
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
vae = LinearVAE(
    num_features=args.num_features,
    latent_dim=args.latent_dim,
    beta=args.beta,
    gamma=args.gamma,
    conditional=args.conditional,
    cond_dim=args.cond_dim,
    use_adversary=args.adversary
).to(device)
probe = LinearProbe(
    vae=None,
    num_features=args.num_features,
    latent_dim=args.latent_dim,
    conditional=args.conditional,
    cond_dim=args.cond_dim
).to(device)

vae_optimizer = optim.Adam(vae.parameters(), lr=args.vae_lr)
probe_optimizer = optim.Adam(probe.parameters(), lr=args.probe_lr)

mse_loss = nn.MSELoss(reduction='sum')

train_dataset = ToyRandomDataset(num_data=int(2e4), num_features=args.num_features)
valid_dataset = ToyRandomDataset(num_data=int(2e3), num_features=args.num_features)
train_queue = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8 if not args.debug else 0
)
valid_queue = DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8 if not args.debug else 0
)

train_writer = SummaryWriter(os.path.join(args.save, 'train'))
valid_writer = SummaryWriter(os.path.join(args.save, 'valid'))

vae_global_step = 0
probe_global_step = 0


def train(epoch):
    vae.train()
    if args.adversary:
        vae.adversary.train()

    i = 0
    global vae_global_step
    for idx, data in tqdm(enumerate(train_queue), desc=f"VAE | Epoch {epoch} | Iterating through train dataset..."):
        data = data.to(device)
        vae_optimizer.zero_grad()
        cond_input = None
        if args.conditional:
            cond_input = data[:, -args.cond_dim:]
        recon, mu, log_var, adv_out = vae(data, cond_input)

        # calculate and propogate vae loss
        recon_loss = vae.recon_loss(recon, data)
        kld_loss = vae.kld_loss(mu, log_var)
        loss = recon_loss + vae.beta * kld_loss
        adv_loss = torch.tensor([0.0])
        if args.adversary:
            adv_loss = vae.recon_loss(adv_out, cond_input)
            loss += vae.gamma * adv_loss

        loss.backward()
        vae_optimizer.step()

        # calculate baseline loss
        bl_recon = 0.5 * torch.ones_like(data)
        bl_recon_loss = mse_loss(bl_recon, data)
        bl_adv_out = 0.5 * torch.ones_like(cond_input)
        bl_adv_loss = mse_loss(bl_adv_out, cond_input)

        # logging
        train_writer.add_scalar('VAE_Loss/recon', recon_loss.item(), vae_global_step)
        train_writer.add_scalar('VAE_Loss/baseline_recon', bl_recon_loss.item(), vae_global_step)
        train_writer.add_scalar('VAE_Loss/kl', kld_loss.item(), vae_global_step)
        train_writer.add_scalar('VAE_Loss/adversarial', adv_loss.item(), vae_global_step)
        train_writer.add_scalar('VAE_Loss/baseline_adversarial', bl_adv_loss.item(), vae_global_step)
        train_writer.add_scalar('VAE_Loss/total', loss.item(), vae_global_step)
        train_writer.add_scalar('Info/beta', vae.beta, vae_global_step)
        train_writer.add_scalar('Info/gamma', vae.gamma, vae_global_step)

        i += 1
        vae_global_step += 1


def test(epoch):
    vae.eval()
    if args.adversary:
        vae.adversary.eval()

    # validation loss
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    total_adv_loss = 0
    total_bl_recon_loss = 0
    total_bl_adv_loss = 0

    '''
    x = (x1, x2), conditional input = x2
    '''
    # perturbation of conditional input loss
    p_c_total_x1_recon_loss = 0
    p_c_total_x2_recon_loss = 0
    p_c_total_x1_bl_recon_loss = 0
    p_c_total_x2_bl_recon_loss = 0

    # perturbation of non-conditional input loss
    p_x1_total_x1_recon_loss = 0
    p_x1_total_x2_recon_loss = 0
    p_x1_total_x1_bl_recon_loss = 0
    p_x1_total_x2_bl_recon_loss = 0

    i = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_queue), desc=f"VAE | Epoch {epoch} | Iterating through valid dataset..."):
            # 1. test original input
            data = data.to(device)
            cond_input = None
            if args.conditional:
                cond_input = data[:, -args.cond_dim:]
            recon, mu, log_var, adv_out = vae(data, cond_input)

            # calculate reconstruction and kl loss
            recon_loss = vae.recon_loss(recon, data)
            kld_loss = vae.kld_loss(mu, log_var)
            loss = recon_loss + vae.beta * kld_loss
            adv_loss = torch.tensor([0.0])
            if args.adversary:
                adv_loss = vae.recon_loss(adv_out, cond_input)
                loss += vae.gamma * adv_loss

            total_loss += loss.item()
            total_kld_loss += kld_loss.item()
            total_recon_loss += recon_loss.item()
            total_adv_loss += adv_loss.item()

            # calculate baseline reconstruction loss
            bl_recon = 0.5 * torch.ones_like(data)
            bl_recon_loss = mse_loss(bl_recon, data)
            bl_adv_out = 0.5 * torch.ones_like(cond_input)
            bl_adv_loss = mse_loss(bl_adv_out, cond_input)
            total_bl_recon_loss += bl_recon_loss.item()
            total_bl_adv_loss += bl_adv_loss.item()

            # log a few samples
            if i == 0:
                sample_input = data[:args.num_samples].cpu().numpy()
                sample_cond_input = None
                if args.conditional:
                    sample_cond_input = cond_input[:args.num_samples].cpu().numpy()
                sample_recon = recon[:args.num_samples].cpu().numpy()
                for s in range(args.num_samples):
                    recon_results = f'Input: {sample_input[s]} ' \
                                    f'Recon: {sample_recon[s]} '
                    if args.conditional:
                        recon_results += f'Conditional Input: {sample_cond_input[s]} '
                    valid_writer.add_text(
                        f'recon/sample_{s}', recon_results, vae_global_step
                    )

            if args.conditional:
                # 2. test perturbing the conditional input c
                p_c_input = torch.rand(data.size(0), args.cond_dim).to(device)
                p_c_recon, *_ = vae(data, p_c_input)

                # calculate reconstruction and kl loss
                p_c_x1_recon_loss = mse_loss(p_c_recon[:, :-args.cond_dim], data[:, :-args.cond_dim])
                p_c_x2_recon_loss = mse_loss(p_c_recon[:, -args.cond_dim:], p_c_input)
                p_c_total_x1_recon_loss += p_c_x1_recon_loss.item()
                p_c_total_x2_recon_loss += p_c_x2_recon_loss.item()

                # calculate baseline reconstruction loss
                p_c_bl_x1_recon = 0.5 * torch.ones_like(data[:, :-args.cond_dim])
                p_c_bl_x1_recon_loss = mse_loss(p_c_bl_x1_recon, data[:, :-args.cond_dim])
                p_c_total_x1_bl_recon_loss += p_c_bl_x1_recon_loss.item()

                p_c_bl_x2_recon = 0.5 * torch.ones_like(p_c_input)
                p_c_bl_x2_recon_loss = mse_loss(p_c_bl_x2_recon, p_c_input)
                p_c_total_x2_bl_recon_loss += p_c_bl_x2_recon_loss.item()

                # log a few samples
                if i == 0:
                    sample_input = data[:args.num_samples].cpu().numpy()
                    sample_cond_input = cond_input[:args.num_samples].cpu().numpy()
                    sample_p_c_input = p_c_input[:args.num_samples].cpu().numpy()
                    sample_recon = recon[:args.num_samples].cpu().numpy()
                    sample_p_recon = p_c_recon[:args.num_samples].cpu().numpy()
                    for s in range(args.num_samples):
                        perturbation_results = f'Input: {sample_input[s]} ' \
                                               f'Original Conditional Input: {sample_cond_input[s]} ' \
                                               f'Perturbed Conditional Input: {sample_p_c_input[s]} ' \
                                               f'Original Recon: {sample_recon[s]} ' \
                                               f'Recon with Perturbed Conditional Input: {sample_p_recon[s]}'
                        valid_writer.add_text(
                            f'perturb_conditional_recon/sample_{s}_perturbation', perturbation_results, vae_global_step
                        )

                # 3. test perturbing the non-conditional input x1
                p_x1_input = torch.rand(data.size(0), args.num_features - args.cond_dim).to(device)
                p_x1_data = torch.cat((p_x1_input, data[:, -args.cond_dim:]), dim=1)
                p_x1_recon, *_ = vae(p_x1_data, cond_input)

                # calculate reconstruction and kl loss
                p_x1_x1_recon_loss = mse_loss(
                    p_x1_recon[:, :-args.cond_dim], p_x1_data[:, :-args.cond_dim]
                )  # reconstruction loss of x1 after perturbing x1
                p_x1_x2_recon_loss = mse_loss(
                    p_x1_recon[:, -args.cond_dim:], cond_input
                )  # reconstruction loss of x2 after perturbing x1
                p_x1_total_x1_recon_loss += p_x1_x1_recon_loss.item()
                p_x1_total_x2_recon_loss += p_x1_x2_recon_loss.item()

                # calculate baseline reconstruction loss
                p_x1_bl_x1_recon = 0.5 * torch.ones_like(data[:, :-args.cond_dim])
                p_x1_bl_x1_recon_loss = mse_loss(p_x1_bl_x1_recon, data[:, :-args.cond_dim])
                p_x1_total_x1_bl_recon_loss += p_x1_bl_x1_recon_loss.item()

                p_x1_bl_x2_recon = 0.5 * torch.ones_like(data[:, -args.cond_dim:])
                p_x1_bl_x2_recon_loss = mse_loss(p_x1_bl_x2_recon, data[:, -args.cond_dim:])
                p_x1_total_x2_bl_recon_loss += p_x1_bl_x2_recon_loss.item()

                # log a few samples
                if i == 0:
                    sample_cond_input = cond_input[:args.num_samples].cpu().numpy()
                    sample_input = data[:args.num_samples].cpu().numpy()
                    sample_p_x1_data = p_x1_data[:args.num_samples].cpu().numpy()
                    sample_recon = recon[:args.num_samples].cpu().numpy()
                    sample_p_x1_recon = p_x1_recon[:args.num_samples].cpu().numpy()
                    for s in range(args.num_samples):
                        perturbation_results = f'Original Input: {sample_input[s]} ' \
                                               f'Perturbed Input: {sample_p_x1_data[s]} ' \
                                               f'Conditional Input: {sample_cond_input[s]} ' \
                                               f'Original Recon: {sample_recon[s]} ' \
                                               f'Recon with Perturbed Non-Conditional Input: {sample_p_x1_recon[s]}'
                        valid_writer.add_text(
                            f'perturb_nonconditional_recon/sample_{s}_perturbation',
                            perturbation_results, vae_global_step
                        )

            # 4. generation through sampling from prior
            if i == 0:
                s_c_input = None  # sample conditional input
                if args.conditional:
                    s_c_input = torch.rand(args.num_samples, args.cond_dim).to(device)
                generation = vae.sample(args.num_samples, s_c_input)
                sample_s_c_input = s_c_input
                if args.conditional:
                    sample_s_c_input = s_c_input.cpu().numpy()
                sample_generation = generation.cpu().numpy()
                for s in range(args.num_samples):
                    generation_results = f'Generated Sample: {sample_generation[s]}'
                    if args.conditional:
                        generation_results = f'Conditional Input: {sample_s_c_input[s]} ' + generation_results
                    valid_writer.add_text(
                        f'generation/sample_{s}_generation', generation_results, vae_global_step
                    )

            i += 1

    # validation logging
    total_loss /= i
    total_kld_loss /= i
    total_recon_loss /= i
    total_adv_loss /= i
    total_bl_recon_loss /= i
    total_bl_adv_loss /= i

    valid_writer.add_scalar('VAE_Loss/recon', total_recon_loss, vae_global_step)
    valid_writer.add_scalar('VAE_Loss/baseline_recon', total_bl_recon_loss, vae_global_step)
    valid_writer.add_scalar('VAE_Loss/kl', total_kld_loss, vae_global_step)
    valid_writer.add_scalar('VAE_Loss/adversarial', total_adv_loss, vae_global_step)
    valid_writer.add_scalar('VAE_Loss/baseline_adversarial', total_bl_adv_loss, vae_global_step)
    valid_writer.add_scalar('VAE_Loss/total', total_loss, vae_global_step)

    # perturb conditional input logging
    p_c_total_x1_recon_loss /= i
    p_c_total_x2_recon_loss /= i
    p_c_total_x1_bl_recon_loss /= i
    p_c_total_x2_bl_recon_loss /= i

    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_x1', p_c_total_x1_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/baseline_recon_x1', p_c_total_x1_bl_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_x2', p_c_total_x2_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/baseline_recon_x2', p_c_total_x2_bl_recon_loss, vae_global_step
    )

    # perturb x1 logging
    p_x1_total_x1_recon_loss /= i
    p_x1_total_x2_recon_loss /= i
    p_x1_total_x1_bl_recon_loss /= i
    p_x1_total_x2_bl_recon_loss /= i

    valid_writer.add_scalar(
        'Perturb_x1_Loss/recon_x1', p_x1_total_x1_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_x1_Loss/baseline_recon_x1', p_x1_total_x1_bl_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_x1_Loss/recon_x2', p_x1_total_x2_recon_loss, vae_global_step
    )
    valid_writer.add_scalar(
        'Perturb_x1_Loss/baseline_recon_x2', p_x1_total_x2_bl_recon_loss, vae_global_step
    )


def train_probe(epoch):
    probe.train()
    probe.vae.eval()
    if args.adversary:
        probe.vae.adversary.eval()

    i = 0
    global probe_global_step
    for idx, data in tqdm(enumerate(train_queue), desc=f"Probe | Epoch {epoch} | Iterating through train dataset..."):
        data = data.to(device)
        probe_optimizer.zero_grad()
        x1_out, x2_out = probe(data)

        # calculate and propogate loss
        x1_gt, x2_gt = data, None
        x2_loss = torch.tensor([0.0])
        if args.conditional:
            x1_gt, x2_gt = data[:, :-args.cond_dim], data[:, -args.cond_dim:]
            x2_loss = mse_loss(x2_out, x2_gt)
        x1_loss = mse_loss(x1_out, x1_gt)
        if args.conditional:
            loss = x1_loss + x2_loss
        else:
            loss = x1_loss

        loss.backward()
        probe_optimizer.step()

        # calculate baseline loss
        x1_bl = 0.5 * torch.ones_like(x1_gt)
        x1_bl_loss = mse_loss(x1_bl, x1_gt)
        x2_bl_loss = torch.tensor([0.0])
        if args.conditional:
            x2_bl = 0.5 * torch.ones_like(x2_gt)
            x2_bl_loss = mse_loss(x2_bl, x2_gt)

        # logging
        train_writer.add_scalar('Probe_Loss/x1', x1_loss.item(), probe_global_step)
        train_writer.add_scalar('Probe_Loss/baseline_x1', x1_bl_loss.item(), probe_global_step)
        train_writer.add_scalar('Probe_Loss/x2', x2_loss.item(), probe_global_step)
        train_writer.add_scalar('Probe_Loss/baseline_x2', x2_bl_loss.item(), probe_global_step)

        i += 1
        probe_global_step += 1


def test_probe(epoch):
    probe.eval()
    probe.vae.eval()
    if args.adversary:
        probe.vae.adversary.eval()

    # validation loss
    total_x1_loss = 0
    total_x2_loss = 0
    total_x1_bl_loss = 0
    total_x2_bl_loss = 0

    i = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_queue),
                              desc=f"Probe | Epoch {epoch} | Iterating through valid dataset..."):
            # 1. test original input
            data = data.to(device)
            x1_out, x2_out = probe(data)

            # calculate loss
            x1_gt, x2_gt = data, None
            if args.conditional:
                x1_gt, x2_gt = data[:, :-args.cond_dim], data[:, -args.cond_dim:]
                x2_loss = mse_loss(x2_out, x2_gt)
                total_x2_loss += x2_loss.item()
            x1_loss = mse_loss(x1_out, x1_gt)
            total_x1_loss += x1_loss.item()

            # calculate baseline loss
            x1_bl = 0.5 * torch.ones_like(x1_gt)
            x1_bl_loss = mse_loss(x1_bl, x1_gt)
            total_x1_bl_loss += x1_bl_loss.item()
            if args.conditional:
                x2_bl = 0.5 * torch.ones_like(x2_gt)
                x2_bl_loss = mse_loss(x2_bl, x2_gt)
                total_x2_bl_loss += x2_bl_loss.item()

            if i == 0:
                # 2. Log a few samples
                sample_x2_gt = None
                sample_x1_gt = data[:args.num_samples].cpu().numpy()
                sample_x2_out = None
                sample_x1_out = x1_out[:args.num_samples].cpu().numpy()
                if args.conditional:
                    sample_x2_gt = data[:args.num_samples, -args.cond_dim:].cpu().numpy()
                    sample_x1_gt = data[:args.num_samples, :-args.cond_dim].cpu().numpy()
                    sample_x2_out = x2_out[:args.num_samples].cpu().numpy()
                for s in range(args.num_samples):
                    probe_results = f'Non-Conditional Ground Truth: {sample_x1_gt[s]} ' \
                                    f'Non-Conditional Output: {sample_x1_out[s]} '
                    if args.conditional:
                        probe_results += f'Conditional Ground Truth: {sample_x2_gt[s]} ' \
                                         f'Conditional Output: {sample_x2_out[s]} '
                    valid_writer.add_text(
                        f'probe_recon/sample_{s}',
                        probe_results, probe_global_step
                    )

            i += 1

    # logging
    total_x1_loss /= i
    total_x2_loss /= i
    total_x1_bl_loss /= i
    total_x2_bl_loss /= i

    valid_writer.add_scalar('Probe_Loss/x1', total_x1_loss, probe_global_step)
    valid_writer.add_scalar('Probe_Loss/baseline_x1', total_x1_bl_loss, probe_global_step)
    valid_writer.add_scalar('Probe_Loss/x2', total_x2_loss, probe_global_step)
    valid_writer.add_scalar('Probe_Loss/baseline_x2', total_x2_bl_loss, probe_global_step)


if __name__ == "__main__":
    # training
    print('Initiating VAE training')
    for epoch in range(1, args.vae_epochs + 1):
        train(epoch)
        test(epoch)

    print('VAE Training: Done.')

    # save last checkpoint
    ckpt_path = os.path.join(args.save, f'checkpoint.pt')
    print(f'Saving the model at {ckpt_path}')
    torch.save(
        {'state_dict': vae.state_dict(), 'optimizer': vae_optimizer.state_dict(), 'global_step': vae_global_step,
         'args': args},
        ckpt_path
    )

    # run probe
    print('Initiating Probe training')
    probe.vae = vae
    probe.vae.eval()
    if args.adversary:
        probe.vae.adversary.eval()

    for epoch in range(1, args.probe_epochs + 1):
        train_probe(epoch)
        test_probe(epoch)

    print('Probe Training: Done.')
