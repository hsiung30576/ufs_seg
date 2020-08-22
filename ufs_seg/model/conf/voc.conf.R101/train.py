from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader
from network import conf
from datasets import VOC

from utils.init_func import init_weight, group_weight
from utils.pyt_utils import all_reduce_tensor
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from legacy.sync_bn import DataParallelModel, Reduce


parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, VOC)

    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)                                       

    BatchNorm2d = nn.BatchNorm2d

    model = conf(config.num_classes, is_training=True,
                    criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model.encoder,
                               BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.spatial_conv,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.context_ff,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.loc_conf,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.refine_block,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.spatial_refine_block,
                               BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.context_refine_block,
                               BatchNorm2d, base_lr * 10)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DataParallelModel(model, device_ids=engine.devices)
    model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            loss = model(imgs, gts)
            loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(2):
                optimizer.param_groups[0]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item()

            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs - 20) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
