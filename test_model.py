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


def move_targets_to_device(targets, device):
    result = []

    for target in targets:
        model_target = {
            key: value.to(device)
            for key, value in target.items()
            if key in MODEL_TARGET_KEYS
        }

        result.append(model_target)

    return result


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用设备：", device)

    train_loader, _, _ = build_data_loaders(
        data_root="./data",
        batch_size=1,
        num_workers=2,
    )

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
    )

    model.to(device)
    model.train()

    images, targets = next(iter(train_loader))

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

    print("\n各部分Loss：")

    for loss_name, loss_value in loss_dict.items():
        print(
            f"{loss_name}: "
            f"{loss_value.item():.6f}"
        )

    print(
        f"\ntotal_loss: "
        f"{total_loss.item():.6f}"
    )

    # 验证反向传播是否正常
    total_loss.backward()

    print("\n前向传播和反向传播均正常。")


if __name__ == "__main__":
    main()