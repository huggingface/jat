# script from:
# github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/codeparrot_training.py

import logging
import os

import time
from argparse import Namespace
from pathlib import Path

from torch.optim import AdamW
from gia.datasets import get_dataloader
import datasets

import transformers
from transformers import get_scheduler, set_seed
from accelerate import Accelerator, DistributedType

from gia.config import Arguments, parse_args
from gia.model import GiaModel


def setup_logging(args: Arguments, accelerator):
    project_name = args.model_ckpt.split("/")[-1]
    logger = logging.getLogger(__name__)
    log_dir = Path(args.save_dir) / "log/"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"debug_{accelerator.process_index}.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / filename), logging.StreamHandler()],
    )
    if accelerator.is_main_process:  # we only want to setup logging once
        accelerator.init_trackers(project_name, vars(args))
        run_name = accelerator.trackers[0].run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name


def create_dataloader(args: Arguments):
    train_dataloader = get_dataloader(
        task_names="mujoco",
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
    )
    return train_dataloader


def get_grouped_params(
    model: GiaModel,
    args: Arguments,
    no_decay=["bias", "ln_1.weight", "ln_2.weight", "ln_f.weight"],
):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def log_metrics(step, metrics, logger, accelerator):
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        accelerator.log(metrics, step)

def main():
    args = parse_args()

    args.use_cache = True  # for debugging
    Arguments.save_args(args)

    # Accelerator
    accelerator = Accelerator(
        log_with=["wandb", "tensorboard"], logging_dir=f"{args.save_dir}/log", dispatch_batches=False
    )
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

    args = Namespace(**vars(args), **acc_state)
    samples_per_step = accelerator.state.num_processes * args.train_batch_size
    set_seed(args.seed)

    # # Clone model repository
    # if accelerator.is_main_process: # TODO
    #     hf_repo = Repository(args.save_dir, clone_from=args.model_ckpt)

    # Logging
    logger, run_name = setup_logging(args, accelerator)
    logger.info(accelerator.state)

    # # Checkout new branch on repo
    # if accelerator.is_main_process and args.push_to_hub: # TODO
    #     hf_repo.git_checkout(run_name, create_branch_ok=True)

    # Load the model
    model = GiaModel(args)

    if args.gradient_checkpointing:
        # TODO: add this to the gia model
        model.gradient_checkpointing_enable()

    # Load dataset and dataloader
    train_dataloader = create_dataloader(args)

    # Prepare the optimizer and learning rate scheduler
    optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    def get_lr():
        # add a scheduler? # TODO
        return optimizer.param_groups[0]["lr"]

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(args.save_dir) if f.is_dir() and "step" in str(f)]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract the step of the checkpoint to continue from there
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("step_", ""))

    # Train model
    model.train()
    device = accelerator.device
    completed_steps = 0
    t_start = time.time()
    loss_tracking = 0
    for step, batch in enumerate(train_dataloader, start=1):
        if args.resume_from_checkpoint and step < resume_step:
            continue  # we need to skip steps until we reach the resumed step

        for k, v in batch.items():
            batch[k] = v.to(device)

        loss = model(batch).loss
        avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        loss_tracking += avg_loss.item() / args.gradient_accumulation_steps
        log_metrics(
            step,
            {"samples": step * samples_per_step, "loss_per_step/train": loss.item()},
            logger,
            accelerator,
        )
        loss = loss / args.gradient_accumulation_steps
        if step % args.gradient_accumulation_steps != 0:
            # Prevent backward from doing gradient all_reduce in every step
            if accelerator.distributed_type == DistributedType.MULTI_GPU:
                with model.no_sync():
                    accelerator.backward(loss)
            else:
                accelerator.backward(loss)
        else:
            lr = get_lr()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)  # should be a hyperparameter
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            elapsed_time = time.time() - t_start
            # tflops = compute_tflops(elapsed_time, accelerator, args)
            log_metrics(
                step,
                {
                    "steps": completed_steps,
                    "loss/train": loss_tracking,
                    "lr": lr,
                    # "tflops": tflops,
                    "time_per_iteration": elapsed_time,
                },
                logger,
                accelerator,
            )
            t_start = time.time()
            loss_tracking = 0
            completed_steps += 1
        if step % args.save_checkpoint_steps == 0:
            accelerator.wait_for_everyone()
            save_dir = os.path.join(args.save_dir, "checkpoints", f"step_{step}")
            accelerator.save_state(save_dir)
            # if accelerator.is_main_process and args.push_to_hub:
            #     hf_repo.push_to_hub(commit_message=f"step {step}")
            model.train()
        if completed_steps >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(args.save_dir, save_function=accelerator.save) # TODO
    save_dir = os.path.join(args.save_dir, f"step_{step}")
    accelerator.save_state(save_dir)
    # if accelerator.is_main_process and args.push_to_hub: # TODO
    #     hf_repo.push_to_hub(commit_message="final model")


if __name__ == "__main__":
    main()
