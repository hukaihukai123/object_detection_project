from dataset_process.data_loader import build_data_loaders


def main():
    train_loader, val_loader, test_loader = (
        build_data_loaders(
            data_root="./data",
            batch_size=2,
            num_workers=0,
        )
    )

    print("训练集图片数：", len(train_loader.dataset))
    print("验证集图片数：", len(val_loader.dataset))
    print("测试集图片数：", len(test_loader.dataset))

    images, targets = next(iter(train_loader))

    print("\n当前Batch图片数：", len(images))

    for index, (image, target) in enumerate(
        zip(images, targets)
    ):
        print(f"\n第{index}张图片")
        print("image：", image.shape, image.dtype)
        print(
            "boxes：",
            target["boxes"].shape,
            target["boxes"].dtype,
        )
        print(
            "labels：",
            target["labels"].shape,
            target["labels"].dtype,
        )
        print("image_id：", target["image_id"])
        print("类别编号：", target["labels"].tolist())
        print("边界框：", target["boxes"])


if __name__ == "__main__":
    main()