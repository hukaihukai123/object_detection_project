import torch
from torch.utils.data import DataLoader

from .build_datasets import build_voc_datasets


def collate_fn(batch):
    """
    目标检测中，不同图片具有：

    1. 不同的宽高
    2. 不同数量的目标

    因此不能使用默认torch.stack。
    返回图片元组和target元组。
    """

    images, targets = zip(*batch)
    return list(images), list(targets)


def build_data_loaders(
    data_root="./data",
    batch_size=2,
    num_workers=0,
):
    train_dataset, val_dataset, test_dataset = (
        build_voc_datasets(data_root)
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader