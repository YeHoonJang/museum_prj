# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

import numpy as np

import os

from . import utils, caption
from .datasets import coco
# from museum.datasets.coco import build_dataset
from .engine import train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device)
    print(f'Initializing Device: {device}')

    # seed 생성
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # captioning 모델 호출
    model, criterion = caption.build_model(args)
    model = torch.nn.DataParallel(model)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # data setting
    dataset_train = coco.build_dataset(args, mode='training')
    dataset_val = coco.build_dataset(args, mode='validation')
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    # checkpoint setting
    if os.path.exists(args.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # Train
    print("Start Training..")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, args.checkpoint)

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        print()

