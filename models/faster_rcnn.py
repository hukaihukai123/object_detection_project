from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
)


def build_faster_rcnn(
    num_classes=21,
    pretrained=True,
):
    """
    num_classes包括背景类别。

    Pascal VOC：
        20个真实类别 + 1个背景类别 = 21
    """

    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
        weights = None

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            num_classes,
        )
    )

    return model