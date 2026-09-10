import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes

from dataset_process.voc_dataset import (
    VOCDataset,
    VOC_CLASSES,
)
from models.faster_rcnn import build_faster_rcnn


def to_uint8(image):
    return (
        image.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
    )


def draw_ground_truth(image, target):
    image = to_uint8(image)
    boxes = target["boxes"].cpu()
    labels = target["labels"].cpu()

    if boxes.numel() == 0:
        return image

    text_labels = [
        VOC_CLASSES[label.item()]
        for label in labels
    ]

    return draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=text_labels,
        colors="red",
        width=3,
        font_size=18,
    )


def draw_prediction(image, prediction, threshold):
    image = to_uint8(image)

    scores = prediction["scores"].cpu()
    keep = scores >= threshold

    boxes = prediction["boxes"].cpu()[keep]
    labels = prediction["labels"].cpu()[keep]
    scores = scores[keep]

    if boxes.numel() == 0:
        return image, 0

    text_labels = [
        f"{VOC_CLASSES[label.item()]} {score.item():.2f}"
        for label, score in zip(labels, scores)
    ]

    drawn = draw_bounding_boxes(
        image=image,
        boxes=boxes,
        labels=text_labels,
        colors="green",
        width=3,
        font_size=18,
    )

    return drawn, len(boxes)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="overfit_test.pth",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="选择8张过拟合图片中的第几张，范围0～7",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--save",
        type=str,
        default="outputs/overfit_prediction.jpg",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    selected_indices = checkpoint["selected_indices"]

    if not 0 <= args.sample < len(selected_indices):
        raise IndexError(
            f"sample应在0～{len(selected_indices) - 1}之间"
        )

    dataset_index = selected_indices[args.sample]

    dataset = VOCDataset(
        root="./data",
        year="2007",
        image_set="train",
        train=False,
        exclude_difficult=True,
        horizontal_flip_prob=0.0,
    )

    image, target = dataset[dataset_index]

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model([
            image.to(device)
        ])[0]

    print("数据集索引：", dataset_index)
    print("真实目标数量：", len(target["boxes"]))

    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()

    keep = scores >= args.threshold

    print("保留的预测数量：", int(keep.sum()))

    for label, score in zip(
        labels[keep],
        scores[keep],
    ):
        print(
            f"{VOC_CLASSES[label.item()]}: "
            f"{score.item():.4f}"
        )

    ground_truth_image = draw_ground_truth(
        image,
        target,
    )

    prediction_image, prediction_count = (
        draw_prediction(
            image,
            prediction,
            args.threshold,
        )
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
    )

    axes[0].imshow(
        ground_truth_image
        .permute(1, 2, 0)
        .numpy()
    )
    axes[0].set_title(
        f"Ground Truth: {len(target['boxes'])} objects"
    )
    axes[0].axis("off")

    axes[1].imshow(
        prediction_image
        .permute(1, 2, 0)
        .numpy()
    )
    axes[1].set_title(
        f"Prediction: {prediction_count} objects"
    )
    axes[1].axis("off")

    plt.tight_layout()

    save_path = Path(args.save)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    print("结果保存到：", save_path)

    plt.show()


if __name__ == "__main__":
    main()