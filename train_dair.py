# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
#
# modified by Junyao Shi
# for training DAIR (Domain and Agent Invariant Representation)
# ---------------------------------------------------------------

import argparse
import time
import torch
import numpy as np
import os
from tqdm import tqdm

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler

from model import AutoEncoder
from thirdparty.adamax import Adamax
import utils
from utils import MASK_THRESHOLD, TYPE_DICT
import datasets

from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3


def flag_check(args):
    args.cond_robot, args.cond_hand = False, False
    # flag logic check
    if args.cond_robot_vec or args.cond_robot_mask:
        assert args.dataset == 'xmagical'
        assert not (args.cond_params_3d or args.cond_hand_bb or args.cond_cropped_hand)
        args.cond_robot = True
    if args.cond_params_3d or args.cond_hand_bb or args.cond_cropped_hand:
        assert args.dataset == 'something-something'
        assert not (args.cond_robot_vec or args.cond_robot_mask)
        args.cond_hand = True
    if args.process_cond_info:
        assert args.cond_robot or args.cond_hand
    if args.zero_latent:
        print('zero_latent mode.')


def generate_cond_info(args, data, x):
    robot_vec, robot_mask = None, None
    params_3d, hand_bb, cropped_hand = None, None, None

    if args.cond_robot_vec:
        robot_vec = torch.cat((data[1], data[3]), dim=1).cuda()
    if args.cond_robot_mask:
        x_clone = torch.clone(x)
        mask = x_clone[:, 0] < MASK_THRESHOLD

        # ignore grey pixels on the edge
        mask[:, 0] = False
        mask[:, -1] = False
        mask[:, :, 0] = False
        mask[:, :, -1] = False

        robot_mask = mask.float().unsqueeze(1).cuda()
    if args.cond_params_3d:
        params_3d = data[1].cuda()
    if args.cond_hand_bb:
        hand_bb = data[2].cuda()
    if args.cond_cropped_hand:
        cropped_hand = data[3].cuda()

    cond_info = None
    if args.cond_robot:
        cond_info = robot_vec, robot_mask
    elif args.cond_hand:
        cond_info = params_3d, hand_bb, cropped_hand

    return cond_info


def generate_cond_info_by_num(args, valid_queue, num_samples):
    robot_vec, robot_mask = None, None
    params_3d, hand_bb, cropped_hand = None, None, None

    if args.cond_robot_vec:
        # we need to feed the encoder info in this case
        # get num_samples sample from valid_queue
        robot_state, robot_type = [], []
        samples_count = 0
        for _, data in enumerate(valid_queue):
            robot_state.append(data[1])
            robot_type.append(data[3])
            samples_count += data[0].size(0)
            if samples_count >= num_samples:
                break
        robot_state = torch.vstack(robot_state)[:num_samples]
        robot_type = torch.vstack(robot_type)[:num_samples]
        robot_vec = torch.cat((robot_state, robot_type), dim=1).cuda()

    if args.cond_robot_mask:
        # we need to feed the encoder info in this case
        # get num_samples sample from valid_queue
        mask = []
        samples_count = 0
        for _, data in enumerate(valid_queue):
            image = data[0]
            image_mask = image[:, 0] < MASK_THRESHOLD

            # ignore grey pixels on the edge
            image_mask[:, 0] = False
            image_mask[:, -1] = False
            image_mask[:, :, 0] = False
            image_mask[:, :, -1] = False

            image_mask = image_mask.float().unsqueeze(1)
            mask.append(image_mask)
            samples_count += data[0].size(0)
            if samples_count >= num_samples:
                break
        mask = torch.vstack(mask)[:num_samples]
        robot_mask = mask.cuda()

    if args.cond_hand:
        params_3d_list, hand_bb_list, cropped_hand_list = [], [], []
        samples_count = 0
        for _, data in enumerate(valid_queue):
            params_3d_list.append(data[1])
            hand_bb_list.append(data[2])
            cropped_hand_list.append(data[3])
            samples_count += data[0].size(0)
            if samples_count >= num_samples:
                break

        if args.cond_params_3d:
            params_3d = torch.vstack(params_3d_list)[:num_samples].cuda()
        if args.cond_hand_bb:
            hand_bb = torch.vstack(hand_bb_list)[:num_samples].cuda()
        if args.cond_cropped_hand:
            cropped_hand = torch.vstack(cropped_hand_list)[:num_samples].cuda()

    cond_info = None
    if args.cond_robot:
        cond_info = robot_vec, robot_mask
    elif args.cond_hand:
        cond_info = params_3d, hand_bb, cropped_hand

    return cond_info


def log_model_output(args, x, cond_info, output, writer, global_step, tag):
    n = int(np.floor(np.sqrt(x.size(0))))
    x_img = x[:n * n]

    output_img_mean = output.mean()
    output_img_sample = output.sample()
    output_img_mean = output_img_mean[:n * n]
    output_img_sample = output_img_sample[:n * n]

    x_tiled = utils.tile_image(x_img, n)
    output_mean_tiled = utils.tile_image(output_img_mean, n)
    output_sample_tiled = utils.tile_image(output_img_sample, n)

    red_line = utils.vertical_red_line(height=x_tiled.size(1), width=3).cuda()
    in_out_mean_tiled = torch.cat((x_tiled, red_line, output_mean_tiled), dim=2)
    in_out_sample_tiled = torch.cat((x_tiled, red_line, output_sample_tiled), dim=2)

    if args.cond_robot_vec:
        robot_vec = cond_info[0][:n * n].cpu().numpy()
        robot_state, robot_type = robot_vec[:, :4], robot_vec[:, 4:].argmax(axis=1).astype(np.uint8)
        robot_state_text, robot_type_text = "", ""
        for n, (rs, rt) in enumerate(zip(robot_state, robot_type)):
            robot_state_text += f'sample {n+1}: robot pos is {rs[:2]} | robot ori is {rs[2:]}  \n'
            robot_type_text += f'sample {n+1}: {TYPE_DICT[rt]}  \n'
        writer.add_text(f'{tag}/robot_state', robot_state_text, global_step)
        writer.add_text(f'{tag}/robot_type', robot_type_text, global_step)
    if args.cond_robot_mask:
        mask_img = cond_info[1][:n * n].repeat(1, 3, 1, 1)
        mask_tiled = utils.tile_image(mask_img, n)
        in_out_mean_tiled = torch.cat((in_out_mean_tiled, red_line, mask_tiled), dim=2)
        in_out_sample_tiled = torch.cat((in_out_sample_tiled, red_line, mask_tiled), dim=2)
    if args.cond_hand_bb:
        hand_bb = cond_info[1][:n * n].cpu().numpy()
        hand_bb_text = ""
        for i, xwyh in enumerate(hand_bb):
            hand_bb_text += f'sample {i + 1}: xwyh is {xwyh}  \n'
        writer.add_text(f'{tag}/hand_bb', hand_bb_text, global_step)
    if args.cond_cropped_hand:
        cropped_hand_img = cond_info[2][:n * n]
        cropped_hand_tiled = utils.tile_image(cropped_hand_img, n)
        in_out_mean_tiled = torch.cat((in_out_mean_tiled, red_line, cropped_hand_tiled), dim=2)
        in_out_sample_tiled = torch.cat((in_out_sample_tiled, red_line, cropped_hand_tiled), dim=2)

    writer.add_image(f'{tag}_mean', in_out_mean_tiled, global_step)
    writer.add_image(f'{tag}_sample', in_out_sample_tiled, global_step)


def log_model_generation(args, model, valid_queue, writer, global_step):
    num_samples = 16
    n = int(np.floor(np.sqrt(num_samples)))

    cond_info = generate_cond_info_by_num(args, valid_queue, num_samples)

    for t in [0.7, 0.8, 0.9, 1.0]:
        logits = model.sample(num_samples, t, cond_info)
        output = model.decoder_output(logits)
        output_img_mean = output.mean()
        output_img_sample = output.sample(t)
        output_mean_tiled = utils.tile_image(output_img_mean, n)
        output_sample_tiled = utils.tile_image(output_img_sample, n)

        red_line = utils.vertical_red_line(height=output_mean_tiled.size(1), width=3).cuda()
        if args.cond_robot_vec:
            robot_vec = cond_info[0][:n * n].cpu().numpy()
            robot_state, robot_type = robot_vec[:, :4], robot_vec[:, 4:].argmax(axis=1).astype(np.uint8)
            robot_state_text, robot_type_text = "", ""
            for i, (rs, rt) in enumerate(zip(robot_state, robot_type)):
                robot_state_text += f'sample {i + 1}: robot pos is {rs[:2]} | robot ori is {rs[2:]}  \n'
                robot_type_text += f'sample {i + 1}: {TYPE_DICT[rt]}  \n'
            writer.add_text('generated_%0.1f/robot_state' % t, robot_state_text, global_step)
            writer.add_text('generated_%0.1f/robot_type' % t, robot_type_text, global_step)
        if args.cond_robot_mask:
            mask_img = cond_info[1][:n * n].repeat(1, 3, 1, 1)
            mask_tiled = utils.tile_image(mask_img, n)
            output_mean_tiled = torch.cat((output_mean_tiled, red_line, mask_tiled), dim=2)
            output_sample_tiled = torch.cat((output_sample_tiled, red_line, mask_tiled), dim=2)
        if args.cond_hand_bb:
            hand_bb = cond_info[1][:n * n].cpu().numpy()
            hand_bb_text = ""
            for i, xwyh in enumerate(hand_bb):
                hand_bb_text += f'sample {i + 1}: xwyh is {xwyh}  \n'
            writer.add_text(f'generated_%0.1f/hand_bb' % t, hand_bb_text, global_step)
        if args.cond_cropped_hand:
            cropped_hand_img = cond_info[2][:n * n]
            cropped_hand_tiled = utils.tile_image(cropped_hand_img, n)
            output_mean_tiled = torch.cat((output_mean_tiled, red_line, cropped_hand_tiled), dim=2)
            output_sample_tiled = torch.cat((output_sample_tiled, red_line, cropped_hand_tiled), dim=2)

        writer.add_image('generated_%0.1f/mean' % t, output_mean_tiled, global_step)
        writer.add_image('generated_%0.1f/sample' % t, output_sample_tiled, global_step)


def log_perturbation(args, model, x, queue, writer, global_step, tag):
    perturbed_data = next(iter(queue))
    perturbed_x = perturbed_data[0] if len(perturbed_data) > 1 else perturbed_data
    perturbed_x = perturbed_x.cuda()

    perturbed_cond_info = generate_cond_info(args, perturbed_data, perturbed_x)

    with torch.no_grad():
        logits, log_q, log_p, kl_all, kl_diag = model(x, perturbed_cond_info)
        output = model.decoder_output(logits)
    log_model_output(args, x, perturbed_cond_info, output, writer, global_step, tag)


def main(args):
    flag_check(args)

    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model = AutoEncoder(args, writer, arch_instance)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2 ** 10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    if args.cont_training:
        checkpoint_name = sorted([f for f in os.listdir(args.save) if 'checkpoint' in f])[-1]
        checkpoint_file = os.path.join(args.save, checkpoint_name)
        # checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
        logging.info('loading the model.')
        logging.info(f'loading checkpoint: {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    for epoch in range(init_epoch, args.epochs):
        # update lrs.
        if args.distributed:
            train_queue.sampler.set_epoch(global_step + args.seed)
            valid_queue.sampler.set_epoch(0)

        if epoch > args.warmup_epochs:
            cnn_scheduler.step()

        # Logging.
        logging.info('epoch %d', epoch)

        # Training.
        train_nelbo, global_step = train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters,
                                         writer, logging, args)
        logging.info('train_nelbo %f', train_nelbo)
        writer.add_scalar('train/nelbo', train_nelbo, global_step)

        model.eval()
        # generate samples less frequently
        eval_freq = 1  # if args.epochs <= 50 else 2  # 20
        if epoch % eval_freq == 0 or epoch == (args.epochs - 1):
            valid_neg_log_p, valid_nelbo = test(
                valid_queue, model, num_samples=10, args=args, logging=logging, global_step=global_step, writer=writer
            )
            logging.info('valid_nelbo %f', valid_nelbo)
            logging.info('valid neg log p %f', valid_neg_log_p)
            logging.info('valid bpd elbo %f', valid_nelbo * bpd_coeff)
            logging.info('valid bpd log p %f', valid_neg_log_p * bpd_coeff)
            writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch)
            writer.add_scalar('val/nelbo', valid_nelbo, epoch)
            writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch)
            writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch)

            # generate samples
            with torch.no_grad():
                log_model_generation(args, model, valid_queue, writer, global_step)

        save_freq = 1 # int(np.ceil(args.epochs / 100))
        if epoch % save_freq == 0 or epoch == (args.epochs - 1):
            if args.global_rank == 0:
                logging.info('saving the model.')
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                            'optimizer': cnn_optimizer.state_dict(), 'global_step': global_step,
                            'args': args, 'arch_instance': arch_instance, 'scheduler': cnn_scheduler.state_dict(),
                            'grad_scalar': grad_scalar.state_dict()},
                           os.path.join(args.save, f'checkpoint.pt'))

    # Final validation
    valid_neg_log_p, valid_nelbo = test(
        valid_queue, model, num_samples=1000, args=args, logging=logging, global_step=global_step, writer=writer
    )
    logging.info('final valid nelbo %f', valid_nelbo)
    logging.info('final valid neg log p %f', valid_neg_log_p)
    writer.add_scalar('val/neg_log_p', valid_neg_log_p, epoch + 1)
    writer.add_scalar('val/nelbo', valid_nelbo, epoch + 1)
    writer.add_scalar('val/bpd_log_p', valid_neg_log_p * bpd_coeff, epoch + 1)
    writer.add_scalar('val/bpd_elbo', valid_nelbo * bpd_coeff, epoch + 1)
    writer.close()


def train(train_queue, model, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging, args):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, data in tqdm(enumerate(train_queue), desc='Going through train data...'):
        if args.print_time:
            start = time.time()
        x = data[0] if len(data) > 1 else data
        x = x.cuda()

        cond_info = generate_cond_info(args, data, x)

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = model(x, cond_info)

            output = model.decoder_output(logits)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)
            kl_coeff *= args.kl_beta  # multiply kl_coeff by beta to simulate beta VAE

            recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(
                    args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if (global_step + 1) % 100 == 0:
            if (global_step + 1) % 100 == 0:  # reduced frequency (originally % 1000)
                log_model_output(args, x, cond_info, output, writer, global_step,
                                 tag='train/reconstruction')
                model.eval()
                log_perturbation(args, model, x, train_queue, writer, global_step,
                                 tag='train/perturbation')
                model.train()

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter',
                              torch.mean(utils.reconstruction_loss(output, x, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1
        if args.print_time:
            end = time.time()
            print(f'Time for this iteration: {end - start}')

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def test(valid_queue, model, num_samples, args, logging, global_step, writer):
    if args.distributed:
        dist.barrier()
    nelbo_avg = utils.AvgrageMeter()
    neg_log_p_avg = utils.AvgrageMeter()
    model.eval()
    for step, data in tqdm(enumerate(valid_queue), 'Going through test data...'):
        x = data[0] if len(data) > 1 else data
        x = x.cuda()

        cond_info = generate_cond_info(args, data, x)

        # change bit length
        x = utils.pre_process(x, args.num_x_bits)

        with torch.no_grad():
            nelbo, log_iw = [], []
            for k in range(num_samples):
                logits, log_q, log_p, kl_all, _ = model(x, cond_info)
                output = model.decoder_output(logits)
                recon_loss = utils.reconstruction_loss(output, x, crop=model.crop_output)
                balanced_kl, _, _ = utils.kl_balancer(kl_all, kl_balance=False)
                nelbo_batch = recon_loss + balanced_kl
                nelbo.append(nelbo_batch)
                log_iw.append(utils.log_iw(output, x, log_q, log_p, crop=model.crop_output))

            nelbo = torch.mean(torch.stack(nelbo, dim=1))
            log_p = torch.mean(torch.logsumexp(torch.stack(log_iw, dim=1), dim=1) - np.log(num_samples))

        nelbo_avg.update(nelbo.data, x.size(0))
        neg_log_p_avg.update(- log_p.data, x.size(0))

        if step == 0:  # reduced frequency
            log_model_output(args, x, cond_info, output, writer, global_step,
                             tag='valid/reconstruction')
            log_perturbation(args, model, x, valid_queue, writer, global_step,
                             tag='valid/perturbation')

    utils.average_tensor(nelbo_avg.avg, args.distributed)
    utils.average_tensor(neg_log_p_avg.avg, args.distributed)
    if args.distributed:
        # block to sync
        dist.barrier()
    logging.info('val, step: %d, NELBO: %f, neg Log p %f', step, nelbo_avg.avg, neg_log_p_avg.avg)
    return neg_log_p_avg.avg, nelbo_avg.avg


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


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
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='This flag enables debug mode, which will set num_worker=0')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64', 'xmagical', 'something-something'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for dataloaders')
    parser.add_argument('--zero_latent', action='store_true', default=False,
                        help='This flag sets the latent sample to the zero tensor for debugging')
    parser.add_argument('--print_time', action='store_true', default=False,
                        help='This flag makes the script prints out time per iteration for tuning batch size')

    parser.add_argument('--cond_robot_vec', action='store_true', default=False,
                        help='This flag enables using robot state and type vector as conditional input of the decoder')
    parser.add_argument('--cond_robot_mask', action='store_true', default=False,
                        help='This flag enables using robot image mask as conditional input of the decoder')
    parser.add_argument('--cond_params_3d', action='store_true', default=False,
                        help='This flag enables using 3D hand pose '
                             'as conditional input of the decoder')
    parser.add_argument('--cond_hand_bb', action='store_true', default=False,
                        help='This flag enables using human hand bounding box information '
                             'as conditional input of the decoder')
    parser.add_argument('--cond_cropped_hand', action='store_true', default=False,
                        help='This flag enables using image of cropped hand '
                             'as conditional input of the decoder')
    parser.add_argument('--process_cond_info', action='store_true', default=False,
                        help='This flag creates a small MLP to process conditional input '
                             'before it is passed into the decoder')


    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    parser.add_argument('--kl_beta', type=float, default=1.0,
                        help='This value simulates beta in beta VAE. '
                             'kl_coeff is multiplied by this value at all times')

    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=10,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--master_port', type=str, default='6020',
                        help='port for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')

    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        # args.distributed = True
        args.distributed = False
        init_processes(0, size, main, args, args.master_port)
