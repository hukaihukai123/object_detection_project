from torch.utils.data import ConcatDataset

from .voc_dataset import VOCDataset


def build_voc_datasets(data_root="./data"):
    # ========================================================
    # 训练集：开启水平翻转，排除difficult目标
    # ========================================================

    voc2007_train = VOCDataset(
        root=data_root,
        year="2007",
        image_set="train",
        train=True,
        exclude_difficult=True,
    )

    voc2012_train = VOCDataset(
        root=data_root,
        year="2012",
        image_set="train",
        train=True,
        exclude_difficult=True,
    )

    train_dataset = ConcatDataset([
        voc2007_train,
        voc2012_train,
    ])

    # ========================================================
    # 验证集：不开启数据增强
    # ========================================================

    voc2007_val = VOCDataset(
        root=data_root,
        year="2007",
        image_set="val",
        train=False,
        exclude_difficult=False,
        horizontal_flip_prob=0.0,
    )

    voc2012_val = VOCDataset(
        root=data_root,
        year="2012",
        image_set="val",
        train=False,
        exclude_difficult=False,
        horizontal_flip_prob=0.0,
    )

    val_dataset = ConcatDataset([
        voc2007_val,
        voc2012_val,
    ])

    # ========================================================
    # 最终测试集
    # ========================================================

    test_dataset = VOCDataset(
        root=data_root,
        year="2007",
        image_set="test",
        train=False,
        exclude_difficult=False,
        horizontal_flip_prob=0.0,
    )

    return train_dataset, val_dataset, test_dataset