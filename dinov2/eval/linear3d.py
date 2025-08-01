# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader, make_classification_dataset_3d
from dinov2.data.transforms import make_classification_transform_3d
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model_3d
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate_dict
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Size of the input images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--dataset-percent",
        type=int,
        help="Percent of finetuning dataset to use",
        default=100
    )
    parser.add_argument(
        "--base-data-dir",
        type=str,
        help="Base data directory for finetuning dataset",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="path to cache directory for monai persistent dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="model name for baseline non-dino comparisons"
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        help="seed for dataset split",
    )
    parser.set_defaults(
        dataset_name='ICBM',
        epochs=10,
        batch_size=128,
        num_workers=8,
        epoch_length=1250,
        save_checkpoint_frequency=20,
        eval_period_iterations=1250,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        classifier_fpath=None,
        image_size=96,
        model_name='',
        dataset_seed=0,
    )
    return parser


def has_ddp_wrapper(m: nn.Module) -> bool:
    return isinstance(m, DistributedDataParallel)


def remove_ddp_wrapper(m: nn.Module) -> nn.Module:
    return m.module if has_ddp_wrapper(m) else m


def _pad_and_collate(batch):
    maxlen = max(len(targets) for image, targets in batch)
    padded_batch = [
        (image, np.pad(targets, (0, maxlen - len(targets)), constant_values=-1)) for image, targets in batch
    ]
    return torch.utils.data.default_collate(padded_batch)


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier):
        super().__init__()
        self.linear_classifier = linear_classifier

    def forward(self, samples, targets):
        preds = self.linear_classifier(samples)
        return {
            "preds": preds,
            "target": targets,
        }


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * distributed.get_global_size()) / 256.0


def setup_linear_classifiers(sample_output, n_last_blocks_list, learning_rates, batch_size, num_classes=1000):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate_linear_classifiers(
    feature_model,
    linear_classifiers,
    data_loader,
    metric,
    metrics_file_path,
    iteration,
    prefixstring="",
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    postprocessors = {k: LinearPostprocessor(v) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    # return prediction dict for test
    if best_classifier_on_val is None:
        _, results_dict_temp = evaluate_dict(
            feature_model,
            data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
        )
    else:
        _, results_dict_temp, pred_dict = evaluate_dict(
            feature_model,
            data_loader,
            postprocessors,
            metrics,
            torch.cuda.current_device(),
            return_preds=True,
        )

    logger.info("")
    results_dict = {}
    max_accuracy = 0
    best_classifier = ""
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
        ) or classifier_string == best_classifier_on_val:
            max_accuracy = metric["top-1"].item()
            best_classifier = classifier_string

    if best_classifier_on_val is None:
        results_dict["best_classifier"] = {
            "name": best_classifier,
            "accuracy": max_accuracy
        }
    else:
        results_dict["best_classifier"] = {
            "name": best_classifier,
            "accuracy": max_accuracy,
            "pred_dict": pred_dict[best_classifier]
        }

    logger.info(f"best classifier: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


def eval_linear(
    *,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of iter, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric,
    resume=True,
    classifier_fpath=None,
):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    max_val_acc = -1

    for data_dict in metric_logger.log_every(
        train_data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data_dict['image'].cuda(non_blocking=True)
        labels = data_dict['label'].cuda(non_blocking=True)

        features = feature_model(data)
        outputs = linear_classifiers(features)

        losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            print("lr", optimizer.param_groups[0]["lr"])

        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if eval_period > 0 and (iteration + 1) % eval_period == 0:
            res_dict = evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric=metric,
                iteration=iteration,
            )
            torch.cuda.synchronize()
            
            # save best model on val to use for test
            if distributed.is_main_process():
                if res_dict["best_classifier"]["accuracy"] > max_val_acc:
                    max_val_acc = res_dict["best_classifier"]["accuracy"]
                    periodic_checkpointer.save("best_val", iteration=iteration)
            torch.cuda.synchronize()

        iteration = iteration + 1

    # load best model
    best_iter = checkpointer.resume_or_load(f'{output_dir}/best_val.pth', resume=False).get("iteration", -1) + 1
    logger.info("Final validation with iter {}".format(best_iter))
    val_results_dict = evaluate_linear_classifiers(
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric=metric,
        iteration=iteration,
    )
    return val_results_dict, feature_model, linear_classifiers, iteration


def test_on_datasets(
    feature_model,
    linear_classifiers,
    test_data_loader,
    metric,
    metrics_file_path,
    iteration,
    best_classifier_on_val,
    prefixstring="",
):
    test_results_dict = evaluate_linear_classifiers(
        feature_model,
        remove_ddp_wrapper(linear_classifiers),
        test_data_loader,
        metric,
        metrics_file_path,
        iteration,
        prefixstring=prefixstring,
        best_classifier_on_val=best_classifier_on_val,
    )
    return test_results_dict


def run_eval_linear(
    model,
    model_name,
    output_dir,
    dataset_name,
    dataset_percent,
    base_data_dir,
    batch_size,
    data_cache_path,
    image_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rates,
    autocast_dtype,
    dataset_seed,
    resume=True,
    classifier_fpath=None,
):
    seed = 0

    # transforms, datasets, metric
    train_transform, val_transform = make_classification_transform_3d(dataset_name, image_size, min_int=-1.0)
    train_dataset, val_dataset, test_dataset, num_classes = make_classification_dataset_3d(
        dataset_name=dataset_name,
        dataset_percent=dataset_percent,
        base_directory=base_data_dir,
        train_transforms=train_transform,
        val_transforms=val_transform,
        cache_path=data_cache_path,
        dataset_seed=dataset_seed,
    )
    metric = build_metric(MetricType.MEAN_ACCURACY, num_classes=num_classes, ks=(1,))

    # linear classifiers
    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    sample_output = feature_model(train_dataset[0]['image'].unsqueeze(0).cuda())
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        n_last_blocks_list,
        learning_rates,
        batch_size,
        num_classes,
    )

    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    max_iter = epochs * epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    # train val test dataloaders
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=SamplerType.SHARDED_INFINITE,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    val_data_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )
    test_data_loader = make_data_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=0,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
        persistent_workers=False,
    )

    checkpoint_period = save_checkpoint_frequency * epoch_length

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
        feature_model=feature_model,
        linear_classifiers=linear_classifiers,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=output_dir,
        max_iter=max_iter,
        checkpoint_period=checkpoint_period,
        running_checkpoint_period=epoch_length,
        eval_period=eval_period_iterations,
        metric=metric,
        resume=resume,
        classifier_fpath=classifier_fpath,
    )

    # load best model
    test_iter = checkpointer.resume_or_load(f'{output_dir}/best_val.pth', resume=False).get("iteration", -1) + 1
    logger.info(f"Testing on {dataset_name}")

    test_results_dict = test_on_datasets(
        feature_model,
        linear_classifiers,
        test_data_loader,
        metric,
        metrics_file_path,
        iteration,
        val_results_dict["best_classifier"]["name"],
        prefixstring="",
    )

    results_dict = {}
    results_dict["best_classifier"] = val_results_dict["best_classifier"]["name"]
    results_dict[f"val_{dataset_name}_accuracy"] = 100.0 * val_results_dict["best_classifier"]["accuracy"]
    results_dict[f"test_{dataset_name}_accuracy"] = 100.0 * test_results_dict["best_classifier"]["accuracy"]
    results_dict[f"test_{dataset_name}_pred_dict"] = test_results_dict["best_classifier"]["pred_dict"]
    logger.info("Test Results Dict " + str(results_dict))

    return results_dict


def main(args):
    model, autocast_dtype = setup_and_build_model_3d(args)

    run_eval_linear(
        model=model,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_percent=args.dataset_percent,
        base_data_dir=args.base_data_dir,
        batch_size=args.batch_size,
        data_cache_path=args.cache_dir,
        image_size=args.image_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        dataset_seed=args.dataset_seed,
    )
    return 0


if __name__ == "__main__":
    description = "DINOv2 3d linear evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
