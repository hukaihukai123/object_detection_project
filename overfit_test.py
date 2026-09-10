import random
from collections import deque

import torch
from torch.utils.data import DataLoader, Subset

from dataset_process.data_loader import collate_fn
from dataset_process.voc_dataset import VOCDataset
from models.faster_rcnn import build_faster_rcnn


MODEL_TARGET_KEYS = {
    "boxes",
    "labels",
    "image_id",
    "area",
    "iscrowd",
}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets, device):
    result = []

    for target in targets:
        result.append({
            key: value.to(device)
            for key, value in target.items()
            if key in MODEL_TARGET_KEYS
        })

    return result


def find_valid_indices(dataset, num_images=8):
    """
    找到包含至少一个普通目标的图片。
    """

    indices = []

    for index in range(len(dataset)):
        _, target = dataset[index]

        if len(target["boxes"]) > 0:
            indices.append(index)

        if len(indices) == num_images:
            break

    if len(indices) < num_images:
        raise RuntimeError(
            f"只找到{len(indices)}张有效图片"
        )

    return indices


def main():
    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用设备：", device)

    # 小样本过拟合时关闭随机增强，
    # 否则模型每次看到的图像不同，更难判断是否成功记忆
    base_dataset = VOCDataset(
        root="./data",
        year="2007",
        image_set="train",
        train=False,
        exclude_difficult=True,
        horizontal_flip_prob=0.0,
    )

    selected_indices = find_valid_indices(
        base_dataset,
        num_images=8,
    )

    print("使用的样本索引：", selected_indices)

    tiny_dataset = Subset(
        base_dataset,
        selected_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    tiny_loader = DataLoader(
        tiny_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=generator,
    )

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
    )

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    max_steps = 150
    current_step = 0

    # 用最近10步的平均Loss观察趋势
    recent_losses = deque(maxlen=10)
    first_average_loss = None

    while current_step < max_steps:
        for images, targets in tiny_loader:
            current_step += 1

            images = [
                image.to(device)
                for image in images
            ]

            targets = move_targets_to_device(
                targets,
                device,
            )

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"第{current_step}步出现无效Loss："
                    f"{total_loss.item()}"
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            recent_losses.append(
                total_loss.item()
            )

            if current_step % 10 == 0:
                average_loss = (
                    sum(recent_losses)
                    / len(recent_losses)
                )

                if first_average_loss is None:
                    first_average_loss = average_loss

                print(
                    f"Step [{current_step:3d}/{max_steps}] "
                    f"average_loss={average_loss:.6f} "
                    f"classifier="
                    f"{loss_dict['loss_classifier'].item():.6f} "
                    f"box="
                    f"{loss_dict['loss_box_reg'].item():.6f} "
                    f"objectness="
                    f"{loss_dict['loss_objectness'].item():.6f} "
                    f"rpn_box="
                    f"{loss_dict['loss_rpn_box_reg'].item():.6f}"
                )

            if current_step >= max_steps:
                break

    final_average_loss = (
        sum(recent_losses)
        / len(recent_losses)
    )

    print("\n训练结束")
    print(
        f"最初平均Loss：{first_average_loss:.6f}"
    )
    print(
        f"最终平均Loss：{final_average_loss:.6f}"
    )

    decrease_ratio = (
        final_average_loss
        / first_average_loss
    )

    print(
        f"最终/最初Loss比例：{decrease_ratio:.3f}"
    )

    if decrease_ratio < 0.5:
        print("小样本过拟合测试通过。")
    else:
        print(
            "Loss下降不够明显，需要继续训练或检查配置。"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "selected_indices": selected_indices,
            "steps": max_steps,
        },
        "overfit_test.pth",
    )

    print("模型已保存：overfit_test.pth")


if __name__ == "__main__":
    main()