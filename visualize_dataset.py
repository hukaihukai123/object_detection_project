import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes

from dataset_process.voc_dataset import (
    VOCDataset,
    VOC_CLASSES,
)


def validate_sample(image, target):
    """
    检查图像和标注格式是否满足Faster R-CNN要求。
    """

    if image.ndim != 3:
        raise ValueError(
            f"image应为[3, H, W]，实际为{image.shape}"
        )

    if image.shape[0] != 3:
        raise ValueError(
            f"image通道数应为3，实际为{image.shape[0]}"
        )

    boxes = target["boxes"]
    labels = target["labels"]

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"boxes应为[N, 4]，实际为{boxes.shape}"
        )

    if labels.ndim != 1:
        raise ValueError(
            f"labels应为[N]，实际为{labels.shape}"
        )

    if len(boxes) != len(labels):
        raise ValueError(
            f"boxes数量{len(boxes)}和labels数量"
            f"{len(labels)}不一致"
        )

    height, width = image.shape[-2:]

    if boxes.numel() > 0:
        xmin = boxes[:, 0]
        ymin = boxes[:, 1]
        xmax = boxes[:, 2]
        ymax = boxes[:, 3]

        if torch.any(xmin < 0) or torch.any(ymin < 0):
            raise ValueError("存在小于0的边界框坐标")

        if torch.any(xmax > width):
            raise ValueError("存在超过图片宽度的xmax")

        if torch.any(ymax > height):
            raise ValueError("存在超过图片高度的ymax")

        if torch.any(xmax <= xmin):
            raise ValueError("存在xmax <= xmin的无效框")

        if torch.any(ymax <= ymin):
            raise ValueError("存在ymax <= ymin的无效框")


def draw_sample(image, target):
    """
    在Tensor图像上绘制边界框和类别名称。

    Returns
    -------
    drawn_image:
        uint8类型的Tensor[3, H, W]
    """

    validate_sample(image, target)

    boxes = target["boxes"].cpu()
    class_ids = target["labels"].cpu()

    difficult = target.get(
        "difficult",
        torch.zeros(
            len(boxes),
            dtype=torch.int64
        )
    ).cpu()

    # image原来是float32且范围为[0, 1]
    # draw_bounding_boxes使用uint8图像更方便
    image_uint8 = (
        image.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
    )

    # 没有目标时直接返回原图
    if boxes.numel() == 0:
        return image_uint8

    text_labels = []
    colors = []

    for class_id, difficult_flag in zip(
        class_ids,
        difficult
    ):
        class_name = VOC_CLASSES[class_id.item()]

        if difficult_flag.item() == 1:
            text_labels.append(
                f"{class_name} [difficult]"
            )
            colors.append("yellow")
        else:
            text_labels.append(class_name)
            colors.append("red")

    drawn_image = draw_bounding_boxes(
        image=image_uint8,
        boxes=boxes,
        labels=text_labels,
        colors=colors,
        width=3,
        font_size=18,
    )

    return drawn_image


def show_tensor_image(
    image,
    title=None,
    save_path=None,
):
    """
    使用matplotlib显示或保存Tensor图像。
    """

    numpy_image = (
        image.permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    plt.imshow(numpy_image)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"可视化结果已保存到：{save_path}")

    plt.show()


def visualize_one(
    dataset,
    index,
    save_path=None,
):
    """
    显示数据集中的一张图像。
    """

    image, target = dataset[index]

    print("=" * 50)
    print(f"数据集索引：{index}")
    print(f"image shape：{image.shape}")
    print(f"image dtype：{image.dtype}")
    print(f"boxes shape：{target['boxes'].shape}")
    print(f"boxes dtype：{target['boxes'].dtype}")
    print(f"labels：{target['labels']}")
    print(f"image_id：{target['image_id']}")
    print(f"difficult：{target['difficult']}")
    print("=" * 50)

    drawn_image = draw_sample(
        image,
        target
    )

    title = (
        f"index={index}, "
        f"objects={len(target['boxes'])}"
    )

    show_tensor_image(
        drawn_image,
        title=title,
        save_path=save_path,
    )


def compare_horizontal_flip(
    normal_dataset,
    flipped_dataset,
    index,
    save_path=None,
):
    """
    对比原图和强制水平翻转后的图片，
    检查边界框是否同步翻转。
    """

    normal_image, normal_target = normal_dataset[index]
    flipped_image, flipped_target = flipped_dataset[index]

    normal_drawn = draw_sample(
        normal_image,
        normal_target
    )

    flipped_drawn = draw_sample(
        flipped_image,
        flipped_target
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    axes[0].imshow(
        normal_drawn.permute(1, 2, 0).numpy()
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        flipped_drawn.permute(1, 2, 0).numpy()
    )
    axes[1].set_title("Horizontal Flip")
    axes[1].axis("off")

    figure.suptitle(
        f"Index={index}: bounding box flip check"
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        print(f"翻转对比图已保存到：{save_path}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Pascal VOC bounding boxes"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--year",
        type=str,
        default="2007",
        choices=["2007", "2012"],
    )

    parser.add_argument(
        "--split",
        type=str,
        default="trainval",
        choices=[
            "train",
            "val",
            "trainval",
            "test",
        ],
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--keep-difficult",
        action="store_true",
        help="显示difficult=1的目标",
    )

    parser.add_argument(
        "--compare-flip",
        action="store_true",
        help="对比原图和强制水平翻转结果",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="可视化图片保存路径"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.year == "2012" and args.split == "test":
        raise ValueError(
            "VOC2012不提供带公开标注的test数据集"
        )

    normal_dataset = VOCDataset(
        root=args.data_root,
        year=args.year,
        image_set=args.split,
        train=False,
        exclude_difficult=not args.keep_difficult,
        horizontal_flip_prob=0.0,
    )

    if not 0 <= args.index < len(normal_dataset):
        raise IndexError(
            f"index={args.index}越界，"
            f"数据集长度为{len(normal_dataset)}"
        )

    if args.compare_flip:
        # 将翻转概率设为1，确保一定发生水平翻转
        flipped_dataset = VOCDataset(
            root=args.data_root,
            year=args.year,
            image_set=args.split,
            train=True,
            exclude_difficult=not args.keep_difficult,
            horizontal_flip_prob=1.0,
        )

        compare_horizontal_flip(
            normal_dataset=normal_dataset,
            flipped_dataset=flipped_dataset,
            index=args.index,
            save_path=args.save,
        )
    else:
        visualize_one(
            dataset=normal_dataset,
            index=args.index,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()