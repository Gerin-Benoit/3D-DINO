# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial
from monai.transforms import Compose, LoadImaged, ScaleIntensityRangePercentilesd, Lambdad
import random
import wandb
import copy

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset_3d, DataAugmentationIJEPA3d
from dinov2.data import collate_data_and_cast, DataAugmentationDINO3d, MaskingGenerator3d, CropForegroundSwapSliceDims, Printer, MaskCollator3D
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup_3d
from dinov2.utils.utils import CosineScheduler, LinearScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch, IJEPAMetaArch

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("3D-JEPA training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="path to cache directory for monai persistent dataset"
    )

    parser.add_argument(
        "--entity",
        default='benoit-gerin',
        type=str,
        help="wandb entity"
    )

    parser.add_argument(
        "--project",
        default='fomo2025',
        type=str,
        help="wandb project"
    )

    parser.add_argument(
        "--mode",
        default='disabled',
        type=str,
        choices=['disabled', 'online'],
        help="wandb mode"
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups)


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.optim["start_lr"],
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.target_encoder["ema_start"],
        final_value=cfg.target_encoder["ema_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )


    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = LinearScheduler(**momentum)

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = wd
        param_group["lr"] = lr


def do_test(cfg, model, iteration):
    new_state_dict = model.target_encoder.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save target encoder checkpoint
        target_encoder_ckp_path = os.path.join(eval_dir, "target_encoder_checkpoint.pth")
        torch.save({"target_encoder": new_state_dict}, target_encoder_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=1250*5,
        max_iter=max_iter,
        max_to_keep=1,
    )


    def random_select_time(x):
        # if time axis exists, select random time slice
        if x.shape[0] > 1:
            t = random.randint(0, x.shape[0] - 1)
            x = x[t:t + 1]
        return x

    # Compose the loading and intensity scaling here to cache transforms in monai persistent dataset
    data_transform = Compose(
            [
                #Printer(),
                LoadImaged(keys=["image"], ensure_channel_first=True),
                Lambdad(keys=["image"], func=random_select_time),
                Lambdad(
                    keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())
                ),  # replace NaNs with mean
                #Printer(),
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
                CropForegroundSwapSliceDims(select_fn=lambda x: x > -1),
                DataAugmentationIJEPA3d(
                    cfg.crops.global_crops_in_slice_scale,
                    cfg.crops.global_crops_cross_slice_scale,
                    global_crops_size=cfg.input.size,
                ),
            ]
        )

    # data collate
    collate_fn = MaskCollator3D(
            input_size=cfg.input.size,
            patch_size=cfg.encoder.patch_size,
            pred_mask_scale=cfg.input.pred_mask_scale,
            enc_mask_scale=cfg.input.enc_mask_scale,
            aspect_ratio= cfg.input.aspect_ratio,
            depth_ratio=cfg.input.depth_ratio,
            nenc= cfg.input.num_enc_masks,
            npred= cfg.input.num_pred_masks,
            allow_overlap_prob=cfg.input.allow_overlap_prob,
            min_keep=cfg.input.min_keep,
        )

    # setup data loader
    dataset = make_dataset_3d(
        dataset_path=cfg.train.dataset_path,
        cache_path=cfg.train.cache_dir,
        data_min_axis_size=cfg.train.data_min_axis_size,
        transform=data_transform
    )
    sampler_type = SamplerType.SHARDED_INFINITE

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    import time
    st = time.time()

    for data in metric_logger.log_every(
        data_loader,
        5,
        header,
        max_iter,
        start_iter,
    ):
        #print(f'batch time: {time.time() - st}')
        current_batch_size = data["collated_batch"].shape[0]
        if iteration > max_iter:
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data)


        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                model.encoder.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                model.encoder.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform target_encoder EMA update
        model.update_target_encoder(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            for k, v in loss_dict.items():
                print(f"loss {k}: {v} NaN:", torch.sum(torch.isnan(v)).item())

            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)


        if distributed.is_main_process():
            wandb.log({"train loss": losses_reduced,
                       "learning rate": lr,
                       "weight decay": wd,
                       "momentum": mom,
                       "current batch size": current_batch_size,
                       })

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

        st = time.time()
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup_3d(args)


    if distributed.is_main_process():
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            entity=args.entity,
            project=args.project,
            # Track hyperparameters and run metadata
            config=args,
            mode=args.mode,
            # name=cfg.LOGGER.TAG
        )


    model = IJEPAMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    #logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)





if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
