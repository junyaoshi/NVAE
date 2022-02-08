import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import AutoEncoder
from probe_model import Probe
import datasets
import utils

MASK_THRESHOLD = 0.91
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()


def main(eval_args):
    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']
    device = torch.device("cuda")

    # check flags
    if not hasattr(args, 'ada_groups'):
        print('old model, no ada groups was found.')
        args.ada_groups = False
    if not hasattr(args, 'min_groups_per_scale'):
        print('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1
    if not hasattr(args, 'num_mixture_dec'):
        print('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    # laod VAE
    print(f'loaded the model at epoch {checkpoint["epoch"]}')
    arch_instance = utils.get_arch_cells(args.arch_instance)
    vae = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    vae.load_state_dict(checkpoint['state_dict'], strict=False)
    vae = vae.to(device)
    vae.eval()
    assert not vae.training

    print('args = %s', args)
    print('vae num conv layers: %d', len(vae.all_conv_layers))
    print('vae param size = %fM ', utils.count_parameters_in_M(vae))

    # initialize probe
    ws_probe = Probe(
        vae=None,
        output_dim=6,
    ).to(device)  # world state probe
    rs_probe = Probe(
        vae=None,
        output_dim=4,
    ).to(device)  # robot state probe
    rt_probe = Probe(
        vae=None,
        output_dim=4,
    ).to(device)  # robot type probe

    # optimizer
    ws_optimizer = optim.Adam(ws_probe.parameters(), lr=eval_args.lr)
    rs_optimizer = optim.Adam(rs_probe.parameters(), lr=eval_args.lr)
    rt_optimizer = optim.Adam(rt_probe.parameters(), lr=eval_args.lr)

    ws_probe.vae = vae
    ws_probe.vae.eval()
    rs_probe.vae = vae
    rs_probe.vae.eval()
    rt_probe.vae = vae
    rt_probe.vae.eval()

    # load train valid queue
    args.data = eval_args.data
    args.batch_size = eval_args.batch_size
    args.debug = eval_args.debug
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)

    # tensorboard writer
    train_writer = SummaryWriter(os.path.join(eval_args.save, 'train'))
    valid_writer = SummaryWriter(os.path.join(eval_args.save, 'valid'))

    global_step, epochs = 0, eval_args.epochs
    for epoch in tqdm(range(epochs), desc=f'Training for {epochs} epochs...'):
        # Training
        global_step = train(
            queue=train_queue,
            probes=[ws_probe, rs_probe, rt_probe],
            optimizers=[ws_optimizer, rs_optimizer, rt_optimizer],
            global_step=global_step,
            writer=train_writer,
            device=device,
            args=args,
            eval_args=eval_args
        )

        # Validation
        global_step = test(
            queue=valid_queue,
            probes=[ws_probe, rs_probe, rt_probe],
            global_step=global_step,
            writer=valid_writer,
            device=device,
            args=args,
            eval_args=eval_args
        )

    print('Probe Training: Done')


def train(queue, probes, optimizers, global_step, writer, device, args, eval_args):
    for probe in probes:
        probe.train()
        probe.vae.eval()
        assert probe.training
        assert not probe.vae.training

    ws_probe, rs_probe, rt_probe = probes
    for step, data in tqdm(enumerate(queue), desc='Going through train data...'):
        x, robot_state, world_state, robot_type = data
        x, robot_state = x.to(device), robot_state.to(device)
        world_state, robot_type = world_state.to(device), robot_type.to(device)

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        for optimizer in optimizers:
            optimizer.zero_grad()

        ws_out = ws_probe(x)
        rs_out = rs_probe(x)
        rt_out = rt_probe(x)

        if eval_args.matching_loss:
            ws_loss_min = torch.inf
            for order in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                a, b, c = order
                indices = torch.tensor([2 * a, 2 * a + 1, 2 * b, 2 * b + 1, 2 * c, 2 * c + 1], device=device)
                ws_out_permuted = torch.index_select(ws_out, 1, indices)
                ws_loss_permuted = mse_loss(ws_out_permuted, world_state)
                if ws_loss_permuted < ws_loss_min:
                    ws_loss_min = ws_loss_permuted
            ws_loss = ws_loss_min
        else:
            ws_loss = mse_loss(ws_out, world_state)
        rs_loss = mse_loss(rs_out, robot_state)
        rt_loss = ce_loss(rt_out, robot_type)

        ws_loss.backward()
        rs_loss.backward()
        rt_loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # calculate baseline losses
        bl_ws = torch.zeros_like(ws_out)
        bl_rs = torch.zeros_like(rs_out)
        bl_rt = F.one_hot(
            torch.tensor([np.random.randint(4, size=rt_out.size(0))], device=device).squeeze(),
            num_classes=4
        ).float()

        if eval_args.matching_loss:
            bl_ws_loss_min = torch.inf
            for order in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                a, b, c = order
                indices = torch.tensor([2 * a, 2 * a + 1, 2 * b, 2 * b + 1, 2 * c, 2 * c + 1], device=device)
                bl_ws_permuted = torch.index_select(bl_ws, 1, indices)
                bl_ws_loss_permuted = mse_loss(bl_ws_permuted, world_state)
                if bl_ws_loss_permuted < bl_ws_loss_min:
                    bl_ws_loss_min = bl_ws_loss_permuted
            bl_ws_loss = bl_ws_loss_min
        else:
            bl_ws_loss = mse_loss(bl_ws, world_state)
        bl_rs_loss = mse_loss(bl_rs, robot_state)
        bl_rt_loss = ce_loss(bl_rt, robot_type)

        # logging
        writer.add_scalar('world_state/loss', ws_loss.item(), global_step)
        writer.add_scalar('world_state/baseline_loss', bl_ws_loss.item(), global_step)
        writer.add_scalar('robot_state/loss', rs_loss.item(), global_step)
        writer.add_scalar('robot_state/baseline_loss', bl_rs_loss.item(), global_step)
        writer.add_scalar('robot_type/loss', rt_loss.item(), global_step)
        writer.add_scalar('robot_type/baseline_loss', bl_rt_loss.item(), global_step)

        global_step += 1
    return global_step


def test(queue, probes, global_step, writer, device, args, eval_args):
    for probe in probes:
        probe.eval()
        probe.vae.eval()
        assert not probe.training
        assert not probe.vae.training

    ws_probe, rs_probe, rt_probe = probes
    for step, data in tqdm(enumerate(queue), desc='Going through valid data...'):
        x, robot_state, world_state, robot_type = data
        x, robot_state = x.to(device), robot_state.to(device)
        world_state, robot_type = world_state.to(device), robot_type.to(device)

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            ws_out = ws_probe(x)
            rs_out = rs_probe(x)
            rt_out = rt_probe(x)

            if eval_args.matching_loss:
                ws_loss_min = torch.inf
                for order in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                    a, b, c = order
                    indices = torch.tensor([2 * a, 2 * a + 1, 2 * b, 2 * b + 1, 2 * c, 2 * c + 1], device=device)
                    ws_out_permuted = torch.index_select(ws_out, 1, indices)
                    ws_loss_permuted = mse_loss(ws_out_permuted, world_state)
                    if ws_loss_permuted < ws_loss_min:
                        ws_loss_min = ws_loss_permuted
                ws_loss = ws_loss_min
            else:
                ws_loss = mse_loss(ws_out, world_state)
            rs_loss = mse_loss(rs_out, robot_state)
            rt_loss = ce_loss(rt_out, robot_type)

            # calculate baseline losses
            bl_ws = torch.zeros_like(ws_out)
            bl_rs = torch.zeros_like(rs_out)
            bl_rt = F.one_hot(
                torch.tensor([np.random.randint(4, size=rt_out.size(0))], device=device).squeeze(),
                num_classes=4
            ).float()

            if eval_args.matching_loss:
                bl_ws_loss_min = torch.inf
                for order in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
                    a, b, c = order
                    indices = torch.tensor([2 * a, 2 * a + 1, 2 * b, 2 * b + 1, 2 * c, 2 * c + 1], device=device)
                    bl_ws_permuted = torch.index_select(bl_ws, 1, indices)
                    bl_ws_loss_permuted = mse_loss(bl_ws_permuted, world_state)
                    if bl_ws_loss_permuted < bl_ws_loss_min:
                        bl_ws_loss_min = bl_ws_loss_permuted
                bl_ws_loss = bl_ws_loss_min
            else:
                bl_ws_loss = mse_loss(bl_ws, world_state)
            bl_rs_loss = mse_loss(bl_rs, robot_state)
            bl_rt_loss = ce_loss(bl_rt, robot_type)

        # logging
        writer.add_scalar('world_state/loss', ws_loss.item(), global_step)
        writer.add_scalar('world_state/baseline_loss', bl_ws_loss.item(), global_step)
        writer.add_scalar('robot_state/loss', rs_loss.item(), global_step)
        writer.add_scalar('robot_state/baseline_loss', bl_rs_loss.item(), global_step)
        writer.add_scalar('robot_type/loss', rt_loss.item(), global_step)
        writer.add_scalar('robot_type/baseline_loss', bl_rt_loss.item(), global_step)

    return global_step


def init_processes(rank, size, fn, args, port='6020'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = port
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='/tmp/expr/checkpoint.pt',
                        help='location of the checkpoint')
    parser.add_argument('--save', type=str, default="logs",
                        help='directory for saving run information')
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train probe (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training probe (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--matching_loss', action='store_true', default=False,
                        help='enables matching loss for world state')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='enables debugging mode, set num_workers=0 to allow for break points')

    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master')
    args = parser.parse_args()
    size = args.num_process_per_node
    init_processes(0, size, main, args, args.master_port)