import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset_process.data_loader import build_data_loaders
from dataset_process.voc_dataset import VOC_CLASSES
from models.faster_rcnn import build_faster_rcnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Faster R-CNN with Pascal VOC metrics"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit evaluated images for a quick smoke test",
    )
    parser.add_argument(
        "--voc07-metric",
        action="store_true",
        help="Use VOC 2007 11-point AP instead of integral AP",
    )
    parser.add_argument(
        "--output",
        default="./outputs/evaluation/metrics.json",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision",
    )
    parser.add_argument("--print-freq", type=int, default=100)
    return parser.parse_args()


def box_iou_one_to_many(box, boxes):
    """Compute IoU between one [4] box and N [N, 4] boxes."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(right - left, 0.0)
        * np.maximum(bottom - top, 0.0)
    )

    box_area = max(box[2] - box[0], 0.0) * max(
        box[3] - box[1], 0.0
    )
    boxes_area = (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    )

    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-12)


def voc_ap(recall, precision, use_07_metric=False):
    """Compute VOC 2007 11-point AP or integral AP."""
    if use_07_metric:
        ap = 0.0
        for threshold in np.arange(0.0, 1.1, 0.1):
            if np.any(recall >= threshold):
                value = np.max(precision[recall >= threshold])
            else:
                value = 0.0
            ap += value / 11.0
        return float(ap)

    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))

    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(
            padded_precision[index],
            padded_precision[index + 1],
        )

    changed = np.where(
        padded_recall[1:] != padded_recall[:-1]
    )[0]

    ap = np.sum(
        (padded_recall[changed + 1] - padded_recall[changed])
        * padded_precision[changed + 1]
    )
    return float(ap)


def collect_predictions(
    model,
    data_loader,
    device,
    amp_enabled,
    score_threshold,
    max_detections,
    max_images,
    print_freq,
):
    """Run inference once and collect CPU predictions and ground truth."""
    model.eval()

    ground_truths = {
        class_id: {}
        for class_id in range(1, len(VOC_CLASSES))
    }
    predictions = {
        class_id: []
        for class_id in range(1, len(VOC_CLASSES))
    }

    processed_images = 0
    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, targets in data_loader:
            if max_images is not None:
                remaining = max_images - processed_images
                if remaining <= 0:
                    break
                images = images[:remaining]
                targets = targets[:remaining]

            device_images = [
                image.to(device, non_blocking=True)
                for image in images
            ]

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(device_images)

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())

                gt_boxes = target["boxes"].detach().cpu().numpy()
                gt_labels = target["labels"].detach().cpu().numpy()
                difficult_tensor = target.get("difficult")
                if difficult_tensor is None:
                    gt_difficult = np.zeros(
                        len(gt_boxes), dtype=np.bool_
                    )
                else:
                    gt_difficult = (
                        difficult_tensor.detach().cpu().numpy().astype(bool)
                    )

                for class_id in range(1, len(VOC_CLASSES)):
                    mask = gt_labels == class_id
                    if np.any(mask):
                        ground_truths[class_id][image_id] = {
                            "boxes": gt_boxes[mask].astype(np.float32),
                            "difficult": gt_difficult[mask],
                        }

                pred_boxes = output["boxes"].detach().cpu().numpy()
                pred_labels = output["labels"].detach().cpu().numpy()
                pred_scores = output["scores"].detach().cpu().numpy()

                keep = np.flatnonzero(pred_scores >= score_threshold)
                keep = keep[:max_detections]

                for index in keep:
                    class_id = int(pred_labels[index])
                    if not 1 <= class_id < len(VOC_CLASSES):
                        continue
                    predictions[class_id].append({
                        "image_id": image_id,
                        "score": float(pred_scores[index]),
                        "box": pred_boxes[index].astype(np.float32),
                    })

                processed_images += 1

            if (
                processed_images == 1
                or processed_images % print_freq == 0
                or (
                    max_images is not None
                    and processed_images >= max_images
                )
            ):
                elapsed = time.perf_counter() - start_time
                speed = processed_images / max(elapsed, 1e-12)
                total = (
                    max_images
                    if max_images is not None
                    else len(data_loader.dataset)
                )
                print(
                    f"Inference [{processed_images}/{total}] "
                    f"{speed:.2f} images/s"
                )

            if max_images is not None and processed_images >= max_images:
                break

    elapsed = time.perf_counter() - start_time
    return ground_truths, predictions, processed_images, elapsed


def evaluate_class(
    class_ground_truths,
    class_predictions,
    iou_threshold,
    use_07_metric,
):
    """Evaluate one class at one IoU threshold with VOC difficult handling."""
    num_positives = sum(
        int(np.sum(~record["difficult"]))
        for record in class_ground_truths.values()
    )

    detections = {
        image_id: np.zeros(len(record["boxes"]), dtype=np.bool_)
        for image_id, record in class_ground_truths.items()
    }

    ranked_predictions = sorted(
        class_predictions,
        key=lambda item: item["score"],
        reverse=True,
    )

    true_positive = []
    false_positive = []

    for prediction in ranked_predictions:
        image_id = prediction["image_id"]
        record = class_ground_truths.get(image_id)

        if record is None:
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue

        overlaps = box_iou_one_to_many(
            prediction["box"],
            record["boxes"],
        )
        best_index = int(np.argmax(overlaps))
        best_overlap = float(overlaps[best_index])

        if best_overlap < iou_threshold:
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue

        if record["difficult"][best_index]:
            # VOC convention: predictions matched to difficult GT are ignored.
            continue

        if not detections[image_id][best_index]:
            true_positive.append(1.0)
            false_positive.append(0.0)
            detections[image_id][best_index] = True
        else:
            # A duplicate prediction for an already detected object is an FP.
            true_positive.append(0.0)
            false_positive.append(1.0)

    true_positive = np.asarray(true_positive, dtype=np.float64)
    false_positive = np.asarray(false_positive, dtype=np.float64)

    if len(true_positive) == 0:
        recall = np.empty((0,), dtype=np.float64)
        precision = np.empty((0,), dtype=np.float64)
        ap = 0.0 if num_positives > 0 else float("nan")
        return {
            "ap": ap,
            "tp": 0,
            "fp": 0,
            "num_positives": num_positives,
            "recall": 0.0,
            "precision": 0.0,
        }

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)

    recall = cumulative_tp / max(num_positives, 1)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-12,
    )

    ap = (
        voc_ap(recall, precision, use_07_metric)
        if num_positives > 0
        else float("nan")
    )

    return {
        "ap": ap,
        "tp": int(cumulative_tp[-1]),
        "fp": int(cumulative_fp[-1]),
        "num_positives": num_positives,
        "recall": float(recall[-1]) if num_positives > 0 else 0.0,
        "precision": float(precision[-1]),
    }


def compute_voc_metrics(
    ground_truths,
    predictions,
    use_07_metric=False,
):
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    maps_by_iou = {}
    ap_by_iou_and_class = defaultdict(dict)

    micro_tp_50 = 0
    micro_fp_50 = 0
    micro_positives_50 = 0

    for iou_threshold in iou_thresholds:
        class_aps = []

        for class_id in range(1, len(VOC_CLASSES)):
            result = evaluate_class(
                ground_truths[class_id],
                predictions[class_id],
                float(iou_threshold),
                use_07_metric,
            )

            class_name = VOC_CLASSES[class_id]
            ap_by_iou_and_class[f"{iou_threshold:.2f}"][class_name] = (
                result["ap"]
            )

            if not np.isnan(result["ap"]):
                class_aps.append(result["ap"])

            if np.isclose(iou_threshold, 0.50):
                micro_tp_50 += result["tp"]
                micro_fp_50 += result["fp"]
                micro_positives_50 += result["num_positives"]

        maps_by_iou[f"{iou_threshold:.2f}"] = (
            float(np.mean(class_aps)) if class_aps else 0.0
        )

    map_50 = maps_by_iou["0.50"]
    map_50_95 = float(np.mean(list(maps_by_iou.values())))

    micro_precision_50 = micro_tp_50 / max(
        micro_tp_50 + micro_fp_50, 1
    )
    micro_recall_50 = micro_tp_50 / max(
        micro_positives_50, 1
    )

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
        "precision_50": float(micro_precision_50),
        "recall_50": float(micro_recall_50),
        "maps_by_iou": maps_by_iou,
        "per_class_ap50": ap_by_iou_and_class["0.50"],
        "per_class_ap_by_iou": dict(ap_by_iou_and_class),
    }


def evaluate_model(
    model,
    data_loader,
    device,
    score_threshold=0.05,
    max_detections=100,
    max_images=None,
    use_07_metric=False,
    amp_enabled=True,
    print_freq=100,
):
    ground_truths, predictions, num_images, inference_seconds = (
        collect_predictions(
            model=model,
            data_loader=data_loader,
            device=device,
            amp_enabled=amp_enabled,
            score_threshold=score_threshold,
            max_detections=max_detections,
            max_images=max_images,
            print_freq=print_freq,
        )
    )

    metrics = compute_voc_metrics(
        ground_truths,
        predictions,
        use_07_metric=use_07_metric,
    )
    metrics.update({
        "num_images": num_images,
        "inference_seconds": float(inference_seconds),
        "images_per_second": float(
            num_images / max(inference_seconds, 1e-12)
        ),
        "score_threshold": float(score_threshold),
        "max_detections": int(max_detections),
        "ap_method": (
            "voc2007_11_point" if use_07_metric else "integral"
        ),
    })
    return metrics


def print_metrics(metrics):
    print("\n" + "=" * 58)
    print(f"Images:        {metrics['num_images']}")
    print(f"mAP@0.50:      {metrics['map_50']:.4f}")
    print(f"mAP@0.50:0.95: {metrics['map_50_95']:.4f}")
    print(f"Precision@0.5: {metrics['precision_50']:.4f}")
    print(f"Recall@0.5:    {metrics['recall_50']:.4f}")
    print(f"Speed:         {metrics['images_per_second']:.2f} images/s")
    print("=" * 58)

    print("\nPer-class AP@0.50")
    for class_name, ap in metrics["per_class_ap50"].items():
        if np.isnan(ap):
            value = "N/A"
        else:
            value = f"{ap:.4f}"
        print(f"{class_name:15s} {value}")


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = device.type == "cuda" and not args.no_amp

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("AMP enabled:", amp_enabled)

    _, val_loader, test_loader = build_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_loader = val_loader if args.split == "val" else test_loader

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)

    print("Checkpoint:", args.checkpoint)
    print("Split:", args.split)
    print("Available images:", len(data_loader.dataset))

    metrics = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        max_images=args.max_images,
        use_07_metric=args.voc07_metric,
        amp_enabled=amp_enabled,
        print_freq=args.print_freq,
    )

    print_metrics(metrics)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\nSaved metrics:", output_path)


if __name__ == "__main__":
    main()
