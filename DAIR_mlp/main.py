import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from model import LinearVAE
from datasets import ToyRandomDataset

parser = argparse.ArgumentParser(description='DAIR MLP')
parser.add_argument('--save', type=str, default="logs",
                    help='directory for saving run information')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10,
                    help='how many batches to wait before printing training status')
parser.add_argument('--num_features', type=int, default=2,
                    help='number of features')
parser.add_argument('--num_samples', type=int, default=8,
                    help='number of samples to generate during testing')
parser.add_argument('--latent_dim', type=int, default=1,
                    help='dimension of VAE latent')
parser.add_argument('--beta', type=float, default=4.0,
                    help='beta value of BetaVAE')
parser.add_argument('--conditional', action='store_true', default=False,
                    help='enables conditional VAE')
parser.add_argument('--cond_dim', type=int, default=1,
                    help='dimension of conditional input (when VAE is conditional)')
parser.add_argument('--debug', action='store_true', default=False,
                    help='enables debugging mode, set num_workers=0 to allow for break points')

args = parser.parse_args()
assert args.num_samples <= args.batch_size
assert args.cond_dim < args.num_features, "conditional input dimension must be less than feature dimension"
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
model = LinearVAE(
    num_features=args.num_features,
    latent_dim=args.latent_dim,
    beta=args.beta,
    conditional=args.conditional,
    cond_dim=args.cond_dim
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()

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

# tensorboard writers
train_writer = SummaryWriter(os.path.join(args.save, 'train'))
valid_writer = SummaryWriter(os.path.join(args.save, 'valid'))
train_bl_writer = SummaryWriter(os.path.join(args.save, 'train_bl'))  # baseline for train set
valid_bl_writer = SummaryWriter(os.path.join(args.save, 'valid_bl'))  # baseline for valid set

global_step = 0


def train(epoch):
    model.train()
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    i = 0
    global global_step
    for idx, data in tqdm(enumerate(train_queue), desc="Iterating through train dataset..."):
        data = data.to(device)
        optimizer.zero_grad()
        cond_input = None
        if args.conditional:
            cond_input = data[:, -args.cond_dim:]
        recon, mu, log_var = model(data, cond_input)

        # calculate and propogate loss
        recon_loss = model.recon_loss(recon, data)
        kld_loss = model.kld_loss(mu, log_var)
        loss = recon_loss + model.beta * kld_loss
        loss.backward()
        optimizer.step()

        # calculate baseline loss
        bl_recon = 0.5 * torch.ones_like(data)
        bl_recon_loss = mse_loss(bl_recon, data)

        # logging
        train_writer.add_scalar('Loss/reconstruction_loss', recon_loss.item(), global_step)
        train_bl_writer.add_scalar('Loss/reconstruction_loss', bl_recon_loss.item(), global_step)
        train_writer.add_scalar('Loss/kl_divergence', kld_loss.item(), global_step)
        train_writer.add_scalar('Loss/total_loss', loss.item(), global_step)
        train_writer.add_scalar('Loss/beta', model.beta, global_step)

        total_loss += loss.item()
        total_kld_loss += kld_loss.item()
        total_recon_loss += recon_loss.item()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * len(data), len(train_dataset),
                       100. * i / len(train_queue),
                       loss.item() / len(data))
            )
        i += 1
        global_step += 1

    total_loss /= i
    total_kld_loss /= i
    total_recon_loss /= i
    print(f'====> Epoch: {epoch} | Average train loss: {total_loss:.4f} | '
          f'Average train KL divergence: {total_kld_loss:.4f} | '
          f'Average train reconstruction loss: {total_recon_loss:.4f}')


def test(epoch):
    model.eval()
    # validation loss
    total_loss = 0
    total_kld_loss = 0
    total_recon_loss = 0
    total_bl_recon_loss = 0

    # perturbation of conditional input loss
    p_cond_total_loss = 0
    p_cond_total_kld_loss = 0
    p_cond_total_orig_recon_loss = 0
    p_cond_total_pert_recon_loss = 0
    p_cond_total_orig_bl_recon_loss = 0
    p_cond_total_pert_bl_recon_loss = 0

    # perturbation of non-conditional input loss
    p_noncond_total_loss = 0
    p_noncond_total_kld_loss = 0
    p_noncond_total_orig_recon_loss = 0
    p_noncond_total_pert_recon_loss = 0
    p_noncond_total_orig_bl_recon_loss = 0
    p_noncond_total_pert_bl_recon_loss = 0
    i = 0
    with torch.no_grad():
        for idx, data in tqdm(enumerate(valid_queue), desc="Iterating through valid dataset..."):
            # 1. test original input
            data = data.to(device)
            cond_input = None
            if args.conditional:
                cond_input = data[:, -args.cond_dim:]
            recon, mu, log_var = model(data, cond_input)

            # calculate reconstruction and kl loss
            recon_loss = model.recon_loss(recon, data)
            kld_loss = model.kld_loss(mu, log_var)
            loss = recon_loss + model.beta * kld_loss
            total_loss += loss.item()
            total_kld_loss += kld_loss.item()
            total_recon_loss += recon_loss.item()

            # calculate baseline reconstruction loss
            bl_recon = 0.5 * torch.ones_like(data)
            bl_recon_loss = mse_loss(bl_recon, data)
            total_bl_recon_loss += bl_recon_loss.item()

            if args.conditional:
                # 2. test perturbed conditional input
                p_cond_input = torch.rand(data.size(0), args.cond_dim).to(device)
                p_cond_recon, p_cond_mu, p_cond_log_var = model(data, p_cond_input)

                # calculate reconstruction and kl loss
                p_cond_orig_recon_loss = mse_loss(p_cond_recon[:, :-args.cond_dim], data[:, :-args.cond_dim])
                p_cond_pert_recon_loss = mse_loss(p_cond_recon[:, -args.cond_dim:], p_cond_input)
                p_cond_recon_loss = p_cond_orig_recon_loss / args.num_features * (args.num_features - args.cond_dim) + \
                                    p_cond_pert_recon_loss / args.num_features * args.cond_dim
                p_cond_kld_loss = model.kld_loss(p_cond_mu, p_cond_log_var)
                p_cond_loss = p_cond_recon_loss + model.beta * p_cond_kld_loss
                p_cond_total_loss += p_cond_loss.item()
                p_cond_total_kld_loss += p_cond_kld_loss.item()
                p_cond_total_orig_recon_loss += p_cond_orig_recon_loss.item()
                p_cond_total_pert_recon_loss += p_cond_pert_recon_loss.item()

                # calculate baseline reconstruction loss
                p_cond_bl_orig_recon = 0.5 * torch.ones_like(data[:, :-args.cond_dim])
                p_cond_bl_orig_recon_loss = mse_loss(p_cond_bl_orig_recon, data[:, :-args.cond_dim])
                p_cond_total_orig_bl_recon_loss += p_cond_bl_orig_recon_loss.item()

                p_cond_bl_pert_recon = 0.5 * torch.ones_like(p_cond_input)
                p_cond_bl_pert_recon_loss = mse_loss(p_cond_bl_pert_recon, p_cond_input)
                p_cond_total_pert_bl_recon_loss += p_cond_bl_pert_recon_loss.item()

                # log a few samples
                if i == 0:
                    sample_input = data[:args.num_samples].cpu().numpy()
                    sample_cond_input = cond_input[:args.num_samples].cpu().numpy()
                    sample_p_cond_input = p_cond_input[:args.num_samples].cpu().numpy()
                    sample_recon = recon[:args.num_samples].cpu().numpy()
                    sample_p_recon = p_cond_recon[:args.num_samples].cpu().numpy()
                    for s in range(args.num_samples):
                        perturbation_results = f'Input: {sample_input[s]} ' \
                                               f'Original Conditional Input: {sample_cond_input[s]} ' \
                                               f'Perturbed Conditional Input: {sample_p_cond_input[s]} ' \
                                               f'Original Recon: {sample_recon[s]} ' \
                                               f'Recon with Perturbed Conditional Input: {sample_p_recon[s]}'
                        valid_writer.add_text(
                            f'perturb_conditional_recon/sample_{s}_perturbation', perturbation_results, global_step
                        )

                # 2. test perturbed non-conditional input
                p_noncond_input = torch.rand(data.size(0), args.num_features - args.cond_dim).to(device)
                p_noncond_data = torch.cat((p_noncond_input, data[:, -args.cond_dim:]), dim=1)
                p_noncond_recon, p_noncond_mu, p_noncond_log_var = model(p_noncond_data, cond_input)

                # calculate reconstruction and kl loss
                p_noncond_pert_recon_loss = mse_loss(
                    p_noncond_recon[:, :-args.cond_dim], p_noncond_data[:, :-args.cond_dim]
                )
                p_noncond_orig_recon_loss = mse_loss(
                    p_noncond_recon[:, -args.cond_dim:], cond_input
                )
                p_noncond_recon_loss = p_noncond_pert_recon_loss / args.num_features \
                                       * (args.num_features - args.cond_dim) + \
                                       p_noncond_orig_recon_loss / args.num_features * args.cond_dim
                p_noncond_kld_loss = model.kld_loss(p_noncond_mu, p_noncond_log_var)
                p_noncond_loss = p_noncond_recon_loss + model.beta * p_noncond_kld_loss
                p_noncond_total_loss += p_noncond_loss.item()
                p_noncond_total_kld_loss += p_noncond_kld_loss.item()
                p_noncond_total_pert_recon_loss += p_noncond_pert_recon_loss.item()
                p_noncond_total_orig_recon_loss += p_noncond_orig_recon_loss.item()

                # calculate baseline reconstruction loss
                p_noncond_bl_pert_recon = 0.5 * torch.ones_like(data[:, :-args.cond_dim])
                p_noncond_bl_pert_recon_loss = mse_loss(p_noncond_bl_pert_recon, data[:, :-args.cond_dim])
                p_noncond_total_pert_bl_recon_loss += p_noncond_bl_pert_recon_loss.item()

                p_noncond_bl_orig_recon = 0.5 * torch.ones_like(data[:, -args.cond_dim:])
                p_noncond_bl_orig_recon_loss = mse_loss(p_noncond_bl_orig_recon, data[:, -args.cond_dim:])
                p_noncond_total_orig_bl_recon_loss += p_noncond_bl_orig_recon_loss.item()

                # log a few samples
                if i == 0:
                    sample_cond_input = cond_input[:args.num_samples].cpu().numpy()
                    sample_input = data[:args.num_samples].cpu().numpy()
                    sample_p_noncond_data = p_noncond_data[:args.num_samples].cpu().numpy()
                    sample_recon = recon[:args.num_samples].cpu().numpy()
                    sample_p_noncond_recon = p_noncond_recon[:args.num_samples].cpu().numpy()
                    for s in range(args.num_samples):
                        perturbation_results = f'Original Input: {sample_input[s]} ' \
                                               f'Perturbed Input: {sample_p_noncond_data[s]} ' \
                                               f'Conditional Input: {sample_cond_input[s]} ' \
                                               f'Original Recon: {sample_recon[s]} ' \
                                               f'Recon with Perturbed Non-Conditional Input: {sample_p_noncond_recon[s]}'
                        valid_writer.add_text(
                            f'perturb_nonconditional_recon/sample_{s}_perturbation', perturbation_results, global_step
                        )

            # 4. generation through sampling from prior
            if i == 0:
                s_cond_input = None
                if args.conditional:
                    s_cond_input = torch.rand(args.num_samples, args.cond_dim).to(device)
                generation = model.sample(args.num_samples, s_cond_input)
                sample_s_cond_input = s_cond_input.cpu().numpy()
                sample_generation = generation.cpu().numpy()
                for s in range(args.num_samples):
                    generation_results = f'Conditional Input: {sample_s_cond_input[s]} ' \
                                         f'Generated Sample: {sample_generation[s]}'
                    valid_writer.add_text(
                        f'generation/sample_{s}_generation', generation_results, global_step
                    )

            i += 1

    # validation logging
    total_loss /= i
    total_kld_loss /= i
    total_recon_loss /= i
    total_bl_recon_loss /= i
    print(f'====> Epoch: {epoch} | Average valid loss: {total_loss:.4f} | '
          f'Average valid KL divergence: {total_kld_loss:.4f} | '
          f'Average valid reconstruction loss: {total_recon_loss:.4f}')
    valid_writer.add_scalar('Loss/reconstruction_loss', total_recon_loss, global_step)
    valid_bl_writer.add_scalar('Loss/reconstruction_loss', total_bl_recon_loss, global_step)
    valid_writer.add_scalar('Loss/kl_divergence', kld_loss.item(), global_step)
    valid_writer.add_scalar('Loss/total_loss', total_loss, global_step)
    valid_writer.add_scalar('Loss/beta', model.beta, global_step)

    # perturb conditional input logging
    p_cond_total_loss /= i
    p_cond_total_kld_loss /= i
    p_cond_total_orig_recon_loss /= i
    p_cond_total_pert_recon_loss /= i
    p_cond_total_orig_bl_recon_loss /= i
    p_cond_total_pert_bl_recon_loss /= i
    p_cond_total_recon_loss = p_cond_total_orig_recon_loss / args.num_features * \
                              (args.num_features - args.cond_dim) + \
                              p_cond_total_pert_recon_loss / args.num_features * args.cond_dim
    p_cond_total_bl_recon_loss = p_cond_total_orig_bl_recon_loss / args.num_features * \
                                 (args.num_features - args.cond_dim) + \
                                 p_cond_total_pert_bl_recon_loss / args.num_features * args.cond_dim
    print(f'====> Epoch: {epoch} | Average perturb loss: {p_cond_total_loss:.4f} | '
          f'Average perturb KL divergence: {p_cond_total_kld_loss:.4f} | '
          f'Average perturb reconstruction loss (original part): {p_cond_total_orig_recon_loss:.4f} | '
          f'Average perturb reconstruction loss (perturbed part): {p_cond_total_pert_recon_loss:.4f} ')
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_original_loss', p_cond_total_orig_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_original_loss', p_cond_total_orig_bl_recon_loss, global_step
    )
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_perturbed_loss', p_cond_total_pert_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Conditional_Loss/recon_perturbed_loss', p_cond_total_pert_bl_recon_loss, global_step
    )
    valid_writer.add_scalar(
        'Perturb_Conditional_Loss/reconstruction_loss', p_cond_total_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Conditional_Loss/reconstruction_loss', p_cond_total_bl_recon_loss, global_step
    )
    valid_writer.add_scalar('Perturb_Conditional_Loss/kl_divergence', p_cond_total_kld_loss, global_step)
    valid_writer.add_scalar('Perturb_Conditional_Loss/total_loss', p_cond_total_loss, global_step)
    valid_writer.add_scalar('Perturb_Conditional_Loss/beta', model.beta, global_step)

    # perturb non-conditional input logging
    p_noncond_total_loss /= i
    p_noncond_total_kld_loss /= i
    p_noncond_total_orig_recon_loss /= i
    p_noncond_total_pert_recon_loss /= i
    p_noncond_total_orig_bl_recon_loss /= i
    p_noncond_total_pert_bl_recon_loss /= i
    p_noncond_total_recon_loss = p_noncond_total_pert_recon_loss / args.num_features * \
                                 (args.num_features - args.cond_dim) + \
                                 p_noncond_total_orig_recon_loss / args.num_features * args.cond_dim
    p_noncond_total_bl_recon_loss = p_noncond_total_pert_bl_recon_loss / args.num_features * \
                                    (args.num_features - args.cond_dim) + \
                                    p_noncond_total_orig_bl_recon_loss / args.num_features * args.cond_dim
    print(f'====> Epoch: {epoch} | Average perturb loss: {p_noncond_total_loss:.4f} | '
          f'Average perturb KL divergence: {p_noncond_total_kld_loss:.4f} | '
          f'Average perturb reconstruction loss (original part): {p_noncond_total_orig_recon_loss:.4f} | '
          f'Average perturb reconstruction loss (perturbed part): {p_noncond_total_pert_recon_loss:.4f} ')
    valid_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/recon_original_loss', p_noncond_total_orig_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/recon_original_loss', p_noncond_total_orig_bl_recon_loss, global_step
    )
    valid_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/recon_perturbed_loss', p_noncond_total_pert_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/recon_perturbed_loss', p_noncond_total_pert_bl_recon_loss, global_step
    )
    valid_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/reconstruction_loss', p_noncond_total_recon_loss, global_step
    )
    valid_bl_writer.add_scalar(
        'Perturb_Non_Conditional_Loss/reconstruction_loss', p_noncond_total_bl_recon_loss, global_step
    )
    valid_writer.add_scalar('Perturb_Non_Conditional_Loss/kl_divergence', p_noncond_total_kld_loss, global_step)
    valid_writer.add_scalar('Perturb_Non_Conditional_Loss/total_loss', p_noncond_total_loss, global_step)
    valid_writer.add_scalar('Perturb_Non_Conditional_Loss/beta', model.beta, global_step)


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
    print('Done.')
