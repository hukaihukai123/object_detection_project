"""Visualize Pascal VOC ground-truth and Faster R-CNN predictions side by side."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch.utils.data import ConcatDataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn

try:
    # Project layout shown by the user: dataset_process/voc_dataset.py
    from dataset_process.voc_dataset import VOCDataset, VOC_CLASSES
except ImportError:
    # Also support placing this script beside voc_dataset.py.
    from voc_dataset import VOCDataset, VOC_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Pascal VOC ground-truth boxes with model predictions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2007", "2012"],
        choices=["2007", "2012"],
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "trainval", "test"],
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Global sample indices. Overrides --num-images.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/prediction_visualizations"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> ConcatDataset:
    if args.split == "test" and args.years != ["2007"]:
        raise ValueError(
            "VOC2012 test annotations are not public. "
            "For test visualization use: --years 2007 --split test"
        )

    datasets = [
        VOCDataset(
            root=args.data_root,
            year=year,
            image_set=args.split,
            train=False,
            exclude_difficult=False,
            horizontal_flip_prob=0.0,
        )
        for year in args.years
    ]
    return ConcatDataset(datasets)


def extract_state_dict(checkpoint: object) -> tuple[dict[str, torch.Tensor], int | None]:
    epoch = None
    if isinstance(checkpoint, dict):
        epoch_value = checkpoint.get("epoch")
        if isinstance(epoch_value, int):
            epoch = epoch_value

        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
        else:
            state_dict = checkpoint
    else:
        raise TypeError("The checkpoint is not a state-dict or checkpoint dictionary.")

    # Support DataParallel and torch.compile checkpoints.
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        cleaned[new_key] = value
    return cleaned, epoch


def build_model(checkpoint_path: Path, device: torch.device):
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=len(VOC_CLASSES),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict, epoch = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, epoch


def locate_sample(dataset: ConcatDataset, global_index: int) -> tuple[str, str]:
    previous_end = 0
    for child, end in zip(dataset.datasets, dataset.cumulative_sizes):
        if global_index < end:
            local_index = global_index - previous_end
            image_id = child.image_ids[local_index]
            return child.year, image_id
        previous_end = end
    raise IndexError(global_index)


def draw_ground_truth(ax, image, target) -> None:
    ax.imshow(image)
    boxes = target["boxes"].cpu()
    labels = target["labels"].cpu()
    difficult = target.get("difficult", torch.zeros(len(boxes), dtype=torch.int64)).cpu()

    for box, label, is_difficult in zip(boxes, labels, difficult):
        x1, y1, x2, y2 = box.tolist()
        color = "#ff9f1c" if int(is_difficult) else "#19b56b"
        linestyle = "--" if int(is_difficult) else "-"
        suffix = " (difficult)" if int(is_difficult) else ""
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
                linestyle=linestyle,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 3),
            f"{VOC_CLASSES[int(label)]}{suffix}",
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1.5, "edgecolor": "none"},
        )

    ax.set_title(f"Ground Truth ({len(boxes)} objects)")
    ax.axis("off")


def draw_predictions(ax, image, prediction, threshold: float) -> int:
    ax.imshow(image)
    keep = prediction["scores"].cpu() >= threshold
    boxes = prediction["boxes"].cpu()[keep]
    labels = prediction["labels"].cpu()[keep]
    scores = prediction["scores"].cpu()[keep]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        color = "#ef3340"
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.2,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 3),
            f"{VOC_CLASSES[int(label)]} {float(score):.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1.5, "edgecolor": "none"},
        )

    ax.set_title(f"Prediction ({len(boxes)} objects, score >= {threshold:.2f})")
    ax.axis("off")
    return len(boxes)


def choose_indices(args: argparse.Namespace, dataset_size: int) -> list[int]:
    if args.indices is not None:
        indices = args.indices
    else:
        if args.num_images <= 0:
            raise ValueError("--num-images must be greater than zero.")
        count = min(args.num_images, dataset_size)
        indices = random.Random(args.seed).sample(range(dataset_size), count)

    invalid = [index for index in indices if not 0 <= index < dataset_size]
    if invalid:
        raise IndexError(f"Indices outside [0, {dataset_size - 1}]: {invalid}")
    return indices


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    dataset = build_dataset(args)
    indices = choose_indices(args, len(dataset))
    model, epoch = build_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint epoch: {epoch if epoch is not None else 'unknown'}")
    print(f"Dataset: VOC {'+'.join(args.years)} {args.split} ({len(dataset)} images)")
    print(f"Selected indices: {indices}")

    for sequence, index in enumerate(indices, start=1):
        image_tensor, target = dataset[index]
        model_input = image_tensor.to(device, non_blocking=True)

        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                prediction = model([model_input])[0]

        image = image_tensor.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        year, image_id = locate_sample(dataset, index)

        height, width = image.shape[:2]
        figure_width = 14
        figure_height = max(5, figure_width * height / (2 * width) + 1.2)
        fig, axes = plt.subplots(1, 2, figsize=(figure_width, figure_height))
        draw_ground_truth(axes[0], image, target)
        prediction_count = draw_predictions(
            axes[1], image, prediction, args.score_threshold
        )
        fig.suptitle(
            f"VOC{year} {image_id} | global index={index}",
            fontsize=13,
        )
        fig.tight_layout()

        output_path = args.output_dir / (
            f"{sequence:02d}_VOC{year}_{image_id}_index{index}.jpg"
        )
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(
            f"[{sequence}/{len(indices)}] {output_path} | "
            f"GT={len(target['boxes'])}, predictions={prediction_count}"
        )

        if args.show:
            plt.show()
        plt.close(fig)

    print(f"Saved visualizations to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
