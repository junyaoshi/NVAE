import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import argparse
from matplotlib import pyplot as plt

import utils
from model import AutoEncoder
import datasets


def main(eval_args):
    # load a checkpoint
    print('loading the model at:')
    print(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        print('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        print('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        print('old model, no num_mixture_dec was found.')
        args.num_mixture_dec = 10

    print(f'loaded the model at epoch {checkpoint["epoch"]}')
    arch_instance = utils.get_arch_cells(args.arch_instance)
    model = AutoEncoder(args, None, arch_instance)
    # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
    # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
    # did not have this variable.
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    print('args = %s', args)
    print('num conv layers: %d', len(model.all_conv_layers))
    print('param size = %fM ', utils.count_parameters_in_M(model))

    # load train valid queue
    args.data = eval_args.data
    args.batch_size = 1
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)

    if eval_args.eval_on_train:
        print('Using the training data for eval.')
        valid_queue = train_queue

    # generate samples
    with torch.no_grad():
        data = next(iter(valid_queue))
        x = data[0] if len(data) > 1 else data
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        info = None
        n = 1
        assert args.cond_robot_state or args.cond_robot_type, "VAE must be conditional"

        # we need to feed the encoder info in this case
        robot_state = data[1]
        robot_type = data[3]
        if args.cond_robot_state and args.cond_robot_type:
            info = torch.cat((robot_state, robot_type), dim=1).cuda()
        elif args.cond_robot_state:
            info = robot_state.cuda()
        elif args.cond_robot_type:
            info = robot_type.cuda()

        logits, log_q, log_p, kl_all, kl_diag = model(x, info)
        output = model.decoder_output(logits)
        output_img = output.mean if isinstance(output,
                                               torch.distributions.bernoulli.Bernoulli) else output.sample()
        x_tiled = utils.tile_image(x, n)
        output_tiled = utils.tile_image(output_img, n)

        # perturb image
        if eval_args.perturb_state:
            assert args.cond_robot_state
            perturbed_robot_pos = 2 * torch.rand(2) - 1
            perturbed_robot_state = torch.clone(robot_state)
            perturbed_robot_state[0, :2] = perturbed_robot_pos
            if args.cond_robot_state and args.cond_robot_type:
                robot_type = data[3]
                info = torch.cat((perturbed_robot_state, robot_type), dim=1).cuda()
            elif args.cond_robot_state:
                info = perturbed_robot_state.cuda()
        else:
            assert args.cond_robot_type
            num_classes = 4
            perturbed_robot_type = F.one_hot(torch.tensor([np.random.randint(num_classes)]),
                                             num_classes=num_classes)
            if args.cond_robot_state and args.cond_robot_type:
                robot_state = data[1]
                info = torch.cat((robot_state, perturbed_robot_type), dim=1).cuda()
            elif args.cond_robot_type:
                info = perturbed_robot_type.cuda()
        perturbed_logits, log_q, log_p, kl_all, kl_diag = model(x, info)
        perturbed_output = model.decoder_output(perturbed_logits)
        perturbed_output_img = output.mean if isinstance(perturbed_output,
                                               torch.distributions.bernoulli.Bernoulli) else perturbed_output.sample()
        perturbed_output_tiled = utils.tile_image(perturbed_output_img, n)


        fig, axs = plt.subplots(3, 1, figsize=(5, 15))
        perturb_type = 'robot_state' if eval_args.perturb_state else 'robot_type'
        original_value = [round(v.item(), 4) for v in robot_state[0, :2]] \
            if eval_args.perturb_state else robot_type.argmax().item()
        perturbed_value = [round(v.item(), 4) for v in perturbed_robot_state[0, :2]] \
            if eval_args.perturb_state else perturbed_robot_type.argmax().item()
        fig.suptitle(f'{perturb_type} perturbation sanity check')
        axs[0].imshow(x_tiled.permute(1, 2, 0).cpu().data.numpy())
        axs[0].set_title(f'Input | {perturb_type}: {original_value}')
        axs[1].imshow(output_tiled.permute(1, 2, 0).cpu().data.numpy())
        axs[1].set_title(f'Output | {perturb_type}: {original_value}')
        axs[2].imshow(perturbed_output_tiled.permute(1, 2, 0).cpu().data.numpy())
        axs[2].set_title(f'Perturbed Output | {perturb_type}: {perturbed_value}')
        plt.tight_layout()
        plt.savefig(f'plots/{perturb_type}_perturb_check_2.png')
        plt.show()
        plt.close()


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
    parser.add_argument('--data', type=str, default='/tmp/data',
                        help='location of the data corpus')
    parser.add_argument('--eval_on_train', action='store_true', default=False,
                        help='Settings this to true will evaluate the model on training data.')

    parser.add_argument('--perturb_state', action='store_true', default=False,
                        help='If true, perturb robot state in the conditional input.')
    parser.add_argument('--perturb_type', action='store_true', default=False,
                        help='If true, perturb robot type in the conditional input.')
    parser.add_argument('--perturb_mask', action='store_true', default=False,
                        help='If true, perturb robot mask in the conditional input.')

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
