import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


# ============================================================
# Pascal VOC类别
# 背景必须编号为0，20个真实类别从1开始
# ============================================================

VOC_CLASSES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

CLASS_TO_IDX = {
    class_name: index
    for index, class_name in enumerate(VOC_CLASSES)
}


# ============================================================
# VOC目标检测数据集
# ============================================================

class VOCDataset(Dataset):
    def __init__(
        self,
        root="./data",
        year="2007",
        image_set="trainval",
        train=True,
        exclude_difficult=True,
        horizontal_flip_prob=0.5,
    ):
        """
        Pascal VOC目标检测数据集。

        Parameters
        ----------
        root:
            数据根目录。

            若设置为"./data"，目录应为：

            data/
            └── VOCdevkit/
                ├── VOC2007/
                └── VOC2012/

        year:
            数据集年份，目前主要使用"2007"或"2012"。

        image_set:
            数据划分：
            "train"、"val"、"trainval"或"test"。

            注意：
            只有VOC2007提供公开的test标注。

        train:
            是否处于训练阶段。
            为True时可以进行随机水平翻转。

        exclude_difficult:
            是否排除XML中difficult=1的目标。

            训练时建议True；
            标准VOC评估时建议False。

        horizontal_flip_prob:
            训练时随机水平翻转的概率。
        """

        super().__init__()

        self.root = Path(root)
        self.year = str(year)
        self.image_set = image_set
        self.train = train
        self.exclude_difficult = exclude_difficult
        self.horizontal_flip_prob = horizontal_flip_prob

        self.voc_root = (
            self.root
            / "VOCdevkit"
            / f"VOC{self.year}"
        )

        self.image_dir = self.voc_root / "JPEGImages"
        self.annotation_dir = self.voc_root / "Annotations"

        self.split_file = (
            self.voc_root
            / "ImageSets"
            / "Main"
            / f"{self.image_set}.txt"
        )

        self._check_directories()
        self.image_ids = self._load_image_ids()

    # --------------------------------------------------------
    # 检查数据目录
    # --------------------------------------------------------

    def _check_directories(self):
        if not self.voc_root.exists():
            raise FileNotFoundError(
                f"找不到VOC数据目录：{self.voc_root}\n"
                f"请检查root和year是否正确。"
            )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"找不到图片目录：{self.image_dir}"
            )

        if not self.annotation_dir.exists():
            raise FileNotFoundError(
                f"找不到标注目录：{self.annotation_dir}"
            )

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"找不到数据划分文件：{self.split_file}"
            )

    # --------------------------------------------------------
    # 读取train、val、trainval或test中的图片编号
    # --------------------------------------------------------

    def _load_image_ids(self):
        with open(
            self.split_file,
            mode="r",
            encoding="utf-8"
        ) as file:
            image_ids = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if len(image_ids) == 0:
            raise RuntimeError(
                f"数据划分文件为空：{self.split_file}"
            )

        return image_ids

    def __len__(self):
        return len(self.image_ids)

    # --------------------------------------------------------
    # 解析单个XML
    # --------------------------------------------------------

    def _parse_xml(
        self,
        annotation_path,
        image_width,
        image_height,
    ):
        """
        将VOC XML转换成边界框和类别编号。

        Returns
        -------
        boxes:
            FloatTensor[N, 4]

        labels:
            Int64Tensor[N]

        difficult:
            Int64Tensor[N]
        """

        tree = ET.parse(annotation_path)
        annotation_root = tree.getroot()

        boxes = []
        labels = []
        difficult_flags = []

        for obj in annotation_root.findall("object"):
            # 读取类别名称
            name_node = obj.find("name")

            if name_node is None or name_node.text is None:
                continue

            class_name = name_node.text.strip().lower()

            if class_name not in CLASS_TO_IDX:
                raise ValueError(
                    f"XML中出现未知类别：{class_name}\n"
                    f"文件：{annotation_path}"
                )

            # 读取difficult
            difficult_node = obj.find("difficult")

            if (
                difficult_node is not None
                and difficult_node.text is not None
            ):
                difficult = int(difficult_node.text)
            else:
                difficult = 0

            if self.exclude_difficult and difficult == 1:
                continue

            # 读取边界框
            bbox_node = obj.find("bndbox")

            if bbox_node is None:
                continue

            xmin_node = bbox_node.find("xmin")
            ymin_node = bbox_node.find("ymin")
            xmax_node = bbox_node.find("xmax")
            ymax_node = bbox_node.find("ymax")

            if any(
                node is None or node.text is None
                for node in [
                    xmin_node,
                    ymin_node,
                    xmax_node,
                    ymax_node,
                ]
            ):
                continue

            xmin = float(xmin_node.text)
            ymin = float(ymin_node.text)
            xmax = float(xmax_node.text)
            ymax = float(ymax_node.text)

            # VOC使用从1开始的闭区间坐标。
            #
            # 例如：
            # xmin=1表示图片的第一个像素。
            #
            # 转换成从0开始的半开区间：
            # [xmin-1, ymin-1, xmax, ymax]
            xmin -= 1.0
            ymin -= 1.0

            # 防止XML标注越过图片边界
            xmin = max(0.0, min(xmin, float(image_width)))
            ymin = max(0.0, min(ymin, float(image_height)))
            xmax = max(0.0, min(xmax, float(image_width)))
            ymax = max(0.0, min(ymax, float(image_height)))

            # Faster R-CNN不接受宽度或高度小于等于0的框
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax,
            ])

            labels.append(
                CLASS_TO_IDX[class_name]
            )

            difficult_flags.append(difficult)

        # 即使没有目标，也必须保证boxes形状为[0, 4]
        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        difficult_flags = torch.as_tensor(
            difficult_flags,
            dtype=torch.int64
        )

        return boxes, labels, difficult_flags

    # --------------------------------------------------------
    # 数据增强
    # --------------------------------------------------------

    def _apply_transforms(self, image, target):
        """
        对图像和边界框同时进行变换。
        """

        # PIL Image转换为FloatTensor[3, H, W]
        # 像素范围由[0, 255]转换成[0, 1]
        image = F.to_tensor(image)

        # 训练阶段随机水平翻转
        if (
            self.train
            and random.random() < self.horizontal_flip_prob
        ):
            image = F.hflip(image)

            image_width = image.shape[-1]

            boxes = target["boxes"].clone()

            if boxes.numel() > 0:
                old_xmin = boxes[:, 0].clone()
                old_xmax = boxes[:, 2].clone()

                boxes[:, 0] = image_width - old_xmax
                boxes[:, 2] = image_width - old_xmin

            target["boxes"] = boxes

        return image, target

    # --------------------------------------------------------
    # 获取一个样本
    # --------------------------------------------------------

    def __getitem__(self, index):
        image_id_string = self.image_ids[index]

        image_path = (
            self.image_dir
            / f"{image_id_string}.jpg"
        )

        annotation_path = (
            self.annotation_dir
            / f"{image_id_string}.xml"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"找不到图片：{image_path}"
            )

        if not annotation_path.exists():
            raise FileNotFoundError(
                f"找不到XML：{annotation_path}"
            )

        image = Image.open(image_path).convert("RGB")

        image_width, image_height = image.size

        boxes, labels, difficult = self._parse_xml(
            annotation_path=annotation_path,
            image_width=image_width,
            image_height=image_height,
        )

        # 计算每个边界框的面积
        if boxes.numel() > 0:
            area = (
                (boxes[:, 2] - boxes[:, 0])
                * (boxes[:, 3] - boxes[:, 1])
            )
        else:
            area = torch.zeros(
                (0,),
                dtype=torch.float32
            )

        # 使2007和2012数据的image_id不重复
        # 例如2007_000027 -> 2007000027
        numeric_image_id = (
        int(self.year) * 1_000_000
        + int(image_id_string)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                numeric_image_id,
                dtype=torch.int64
            ),
            "area": area,
            "iscrowd": torch.zeros(
                (len(boxes),),
                dtype=torch.int64
            ),

            # Faster R-CNN不使用这个字段，
            # 但后续标准VOC评估需要它
            "difficult": difficult,
        }

        image, target = self._apply_transforms(
            image,
            target
        )

        return image, target


# ============================================================
# 单文件测试
# ============================================================

if __name__ == "__main__":
    dataset = VOCDataset(
        root="./data",
        year="2007",
        image_set="trainval",
        train=False,
        exclude_difficult=True,
    )

    image, target = dataset[0]

    print("数据集长度：", len(dataset))
    print("图像形状：", image.shape)
    print("图像类型：", image.dtype)
    print("边界框：", target["boxes"])
    print("边界框形状：", target["boxes"].shape)
    print("标签：", target["labels"])
    print("标签类型：", target["labels"].dtype)
    print("图像编号：", target["image_id"])
    print("目标面积：", target["area"])
    print("困难目标：", target["difficult"])