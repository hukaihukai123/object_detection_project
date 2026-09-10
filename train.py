import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch

from dataset_process.data_loader import build_data_loaders
from models.faster_rcnn import build_faster_rcnn


MODEL_TARGET_KEYS = {
    "boxes",
    "labels",
    "image_id",
    "area",
    "iscrowd",
}

LOSS_NAMES = (
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on Pascal VOC"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--snapshot-freq", type=int, default=5)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Limit batches per epoch for a smoke test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path, for example outputs/checkpoints/latest.pth",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(images, targets, device):
    images = [
        image.to(device, non_blocking=True)
        for image in images
    ]

    model_targets = []
    for target in targets:
        model_targets.append({
            key: value.to(device, non_blocking=True)
            for key, value in target.items()
            if key in MODEL_TARGET_KEYS
        })

    return images, model_targets


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    scaler,
    device,
    epoch,
    amp_enabled,
    print_freq,
    max_steps_per_epoch,
):
    model.train()

    running = {
        "total_loss": 0.0,
        **{name: 0.0 for name in LOSS_NAMES},
    }

    num_batches = len(data_loader)
    if max_steps_per_epoch is not None:
        num_batches = min(num_batches, max_steps_per_epoch)
    epoch_start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, (images, targets) in enumerate(data_loader, start=1):
        images, targets = move_batch_to_device(
            images,
            targets,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

        if not torch.isfinite(total_loss):
            detached_losses = {
                name: float(value.detach().cpu())
                for name, value in loss_dict.items()
            }
            raise RuntimeError(
                f"Non-finite loss at epoch={epoch + 1}, step={step}: "
                f"{detached_losses}"
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running["total_loss"] += total_loss.detach().item()
        for name in LOSS_NAMES:
            running[name] += loss_dict[name].detach().item()

        if step == 1 or step % print_freq == 0 or step == num_batches:
            elapsed = time.perf_counter() - epoch_start
            seconds_per_step = elapsed / step
            remaining_minutes = (
                seconds_per_step * (num_batches - step) / 60.0
            )

            print(
                f"Epoch [{epoch + 1}] "
                f"Step [{step}/{num_batches}] "
                f"loss={running['total_loss'] / step:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"eta={remaining_minutes:.1f} min"
            )

        if step >= num_batches:
            break

    epoch_seconds = time.perf_counter() - epoch_start
    averages = {
        name: value / num_batches
        for name, value in running.items()
    }

    if device.type == "cuda":
        averages["peak_memory_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
    else:
        averages["peak_memory_gb"] = 0.0

    averages["epoch_seconds"] = epoch_seconds
    return averages


def atomic_save(checkpoint, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def make_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    args,
):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }


def append_csv(log_path, row):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("AMP enabled:", amp_enabled)
    print("Seed:", args.seed)

    train_loader, _, _ = build_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Training images:", len(train_loader.dataset))
    print("Batches per epoch:", len(train_loader))

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[8, 12],
        gamma=0.1,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_path = output_dir / "logs" / "train_log.csv"

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            amp_enabled=amp_enabled,
            print_freq=args.print_freq,
            max_steps_per_epoch=args.max_steps_per_epoch,
        )

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "total_loss": metrics["total_loss"],
            "loss_classifier": metrics["loss_classifier"],
            "loss_box_reg": metrics["loss_box_reg"],
            "loss_objectness": metrics["loss_objectness"],
            "loss_rpn_box_reg": metrics["loss_rpn_box_reg"],
            "epoch_seconds": metrics["epoch_seconds"],
            "peak_memory_gb": metrics["peak_memory_gb"],
        }
        append_csv(log_path, row)

        checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
        )

        latest_path = checkpoint_dir / "latest.pth"
        atomic_save(checkpoint, latest_path)

        if (
            (epoch + 1) % args.snapshot_freq == 0
            or epoch + 1 == args.epochs
        ):
            snapshot_path = (
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            )
            atomic_save(checkpoint, snapshot_path)

        print(
            f"Epoch {epoch + 1} finished: "
            f"loss={metrics['total_loss']:.4f}, "
            f"time={metrics['epoch_seconds'] / 60:.1f} min, "
            f"peak_memory={metrics['peak_memory_gb']:.2f} GB"
        )
        print("Saved:", latest_path)

    print("Training finished.")
    print("Log:", log_path)


if __name__ == "__main__":
    main()
