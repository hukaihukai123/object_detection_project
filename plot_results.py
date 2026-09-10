"""Create presentation-ready Faster R-CNN training and evaluation figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

LOSS_COLUMNS = {
    "loss_classifier": "Classifier",
    "loss_box_reg": "Box regression",
    "loss_objectness": "RPN objectness",
    "loss_rpn_box_reg": "RPN box regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Faster R-CNN training curves and Pascal VOC AP results."
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        default=Path("./outputs/fasterrcnn_voc/training_log.csv"),
        help="CSV produced by train.py.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("./outputs/evaluation/metrics.json"),
        help="JSON produced by evaluator.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/report_figures"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#1e293b",
            "axes.titlecolor": "#0f172a",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#dbe3ec",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "legend.frameon": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "savefig.facecolor": "white",
        }
    )


def read_training_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"Training CSV is empty: {path}")

    required = {"epoch", "learning_rate", "total_loss", *LOSS_COLUMNS}
    missing = required.difference(rows[0])
    if missing:
        raise KeyError(f"Training CSV is missing columns: {sorted(missing)}")

    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in required
    }


def normalized_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def find_value(data: Any, candidate_keys: set[str]) -> Any | None:
    """Recursively find the first value whose normalized key matches."""
    normalized_candidates = {normalized_key(key) for key in candidate_keys}
    if isinstance(data, dict):
        for key, value in data.items():
            if normalized_key(str(key)) in normalized_candidates:
                return value
        for value in data.values():
            result = find_value(value, candidate_keys)
            if result is not None:
                return result
    return None


def metric_number(data: dict[str, Any], keys: set[str]) -> float | None:
    value = find_value(data, keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_evaluation_json(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        raw = json.load(file)

    per_class_raw = find_value(
        raw,
        {
            "per_class_ap_50",
            "per_class_ap50",
            "per_class_ap@0.50",
            "ap50_per_class",
            "class_ap50",
        },
    )

    if isinstance(per_class_raw, dict):
        per_class = {
            class_name: float(per_class_raw[class_name])
            for class_name in VOC_CLASSES
            if class_name in per_class_raw
        }
    elif isinstance(per_class_raw, list) and len(per_class_raw) == len(VOC_CLASSES):
        per_class = dict(zip(VOC_CLASSES, map(float, per_class_raw)))
    else:
        # Support JSON files that store the 20 class AP values at top level.
        per_class = {
            class_name: float(raw[class_name])
            for class_name in VOC_CLASSES
            if class_name in raw and isinstance(raw[class_name], (int, float))
        }

    if len(per_class) != len(VOC_CLASSES):
        missing = sorted(set(VOC_CLASSES).difference(per_class))
        raise KeyError(
            "Could not find all per-class AP@0.50 values in metrics JSON. "
            f"Missing: {missing}"
        )

    metrics = {
        "mAP@0.50": metric_number(raw, {"map_50", "map50", "mAP@0.50"}),
        "mAP@0.50:0.95": metric_number(
            raw, {"map", "map_50_95", "map5095", "mAP@0.50:0.95"}
        ),
        "Precision@0.50": metric_number(
            raw, {"precision_50", "precision50", "precision@0.5"}
        ),
        "Recall@0.50": metric_number(raw, {"recall_50", "recall50", "recall@0.5"}),
        "Speed (img/s)": metric_number(
            raw, {"speed", "images_per_second", "fps", "inference_speed"}
        ),
        "Images": metric_number(raw, {"images", "num_images", "image_count"}),
    }

    # The per-class mean is a reliable fallback for AP@0.50.
    if metrics["mAP@0.50"] is None:
        metrics["mAP@0.50"] = float(np.mean(list(per_class.values())))

    return metrics, per_class


def add_lr_change_lines(ax, epochs: np.ndarray, learning_rates: np.ndarray) -> None:
    changes = np.flatnonzero(learning_rates[1:] != learning_rates[:-1]) + 1
    for position in changes:
        epoch = epochs[position]
        ax.axvline(epoch, color="#94a3b8", linestyle="--", linewidth=1.1)
        ax.text(
            epoch + 0.12,
            ax.get_ylim()[1] * 0.97,
            f"lr={learning_rates[position]:g}",
            color="#64748b",
            fontsize=8,
            va="top",
        )


def save_total_loss(training: dict[str, np.ndarray], output: Path, dpi: int) -> None:
    epochs = training["epoch"]
    losses = training["total_loss"]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(
        epochs,
        losses,
        color="#2563eb",
        marker="o",
        markersize=4.8,
        linewidth=2.4,
        label="Total loss",
    )
    ax.fill_between(epochs, losses, alpha=0.10, color="#2563eb")
    add_lr_change_lines(ax, epochs, training["learning_rate"])
    ax.set_title("Faster R-CNN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_xticks(epochs.astype(int))
    ax.set_xlim(epochs.min(), epochs.max())
    ax.legend(loc="lower left")
    reduction = (1.0 - losses[-1] / losses[0]) * 100.0
    ax.text(
        0.99,
        0.78,
        f"{losses[0]:.4f} -> {losses[-1]:.4f}\nReduction: {reduction:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(output / "01_total_loss.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_component_losses(
    training: dict[str, np.ndarray], output: Path, dpi: int
) -> None:
    colors = ["#7c3aed", "#dc2626", "#059669", "#ea580c"]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    for (column, label), color in zip(LOSS_COLUMNS.items(), colors):
        ax.plot(
            training["epoch"],
            training[column],
            marker="o",
            markersize=3.8,
            linewidth=2.0,
            label=label,
            color=color,
        )
    ax.set_title("Training Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average loss")
    ax.set_xticks(training["epoch"].astype(int))
    ax.set_xlim(training["epoch"].min(), training["epoch"].max())
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "02_loss_components.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_ap_chart(
    metrics: dict[str, float | None],
    per_class: dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    ordered = sorted(per_class.items(), key=lambda item: item[1])
    names = [item[0] for item in ordered]
    values = np.asarray([item[1] for item in ordered]) * 100.0
    map_50 = float(metrics["mAP@0.50"]) * 100.0

    colors = [
        "#dc2626" if value < 60 else "#f59e0b" if value < 75 else "#2563eb"
        for value in values
    ]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    bars = ax.barh(names, values, color=colors, height=0.68)
    ax.axvline(
        map_50,
        color="#111827",
        linestyle="--",
        linewidth=1.5,
        label=f"mAP@0.50 = {map_50:.2f}%",
    )
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
    ax.set_title("VOC2007 Test AP@0.50 by Class")
    ax.set_xlabel("Average precision (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "03_ap50_by_class.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_dashboard(
    training: dict[str, np.ndarray],
    metrics: dict[str, float | None],
    per_class: dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.05, 1.35], height_ratios=[1, 1], wspace=0.22, hspace=0.30
    )

    ax_loss = fig.add_subplot(grid[0, 0])
    ax_loss.plot(
        training["epoch"],
        training["total_loss"],
        color="#2563eb",
        marker="o",
        linewidth=2.4,
    )
    ax_loss.fill_between(
        training["epoch"], training["total_loss"], color="#2563eb", alpha=0.10
    )
    ax_loss.set_title("Training convergence")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Total loss")
    ax_loss.set_xticks(training["epoch"].astype(int))

    ax_cards = fig.add_subplot(grid[1, 0])
    ax_cards.axis("off")
    card_items = [
        ("mAP@0.50", metrics.get("mAP@0.50"), "%", 100.0),
        ("mAP@0.50:0.95", metrics.get("mAP@0.50:0.95"), "%", 100.0),
        ("Recall@0.50", metrics.get("Recall@0.50"), "%", 100.0),
        ("Inference speed", metrics.get("Speed (img/s)"), " img/s", 1.0),
    ]
    positions = [(0.02, 0.58), (0.52, 0.58), (0.02, 0.08), (0.52, 0.08)]
    for (label, value, suffix, scale), (x, y) in zip(card_items, positions):
        display = "N/A" if value is None else f"{value * scale:.2f}{suffix}"
        ax_cards.text(
            x,
            y,
            f"{display}\n{label}",
            transform=ax_cards.transAxes,
            fontsize=19,
            fontweight="bold",
            color="#0f172a",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
            },
        )

    ax_ap = fig.add_subplot(grid[:, 1])
    ordered = sorted(per_class.items(), key=lambda item: item[1])
    names = [item[0] for item in ordered]
    values = np.asarray([item[1] for item in ordered]) * 100.0
    map_50 = float(metrics["mAP@0.50"]) * 100.0
    colors = ["#f59e0b" if value < 75 else "#2563eb" for value in values]
    bars = ax_ap.barh(names, values, color=colors, height=0.67)
    ax_ap.axvline(map_50, color="#111827", linestyle="--", linewidth=1.5)
    ax_ap.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
    ax_ap.set_title("VOC2007 test: AP@0.50")
    ax_ap.set_xlabel("Average precision (%)")
    ax_ap.set_xlim(0, 100)
    ax_ap.grid(axis="y", visible=False)

    fig.suptitle(
        "Faster R-CNN ResNet50-FPN | Pascal VOC Detection Results",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
        y=0.985,
    )
    fig.savefig(output / "04_experiment_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_markdown_summary(
    training: dict[str, np.ndarray],
    metrics: dict[str, float | None],
    per_class: dict[str, float],
    output: Path,
) -> None:
    best_classes = sorted(per_class.items(), key=lambda item: item[1], reverse=True)[:5]
    weak_classes = sorted(per_class.items(), key=lambda item: item[1])[:5]

    def percent(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def number(value: float | None, decimals: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{decimals}f}"

    lines = [
        "# Faster R-CNN experiment summary",
        "",
        "## Main results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| mAP@0.50 | {percent(metrics.get('mAP@0.50'))} |",
        f"| mAP@0.50:0.95 | {percent(metrics.get('mAP@0.50:0.95'))} |",
        f"| Precision@0.50 | {percent(metrics.get('Precision@0.50'))} |",
        f"| Recall@0.50 | {percent(metrics.get('Recall@0.50'))} |",
        f"| Inference speed | {number(metrics.get('Speed (img/s)'))} images/s |",
        f"| Test images | {number(metrics.get('Images'), 0)} |",
        "",
        "## Training convergence",
        "",
        f"Total loss decreased from {training['total_loss'][0]:.4f} "
        f"to {training['total_loss'][-1]:.4f} over {len(training['epoch'])} epochs, "
        f"a reduction of {(1 - training['total_loss'][-1] / training['total_loss'][0]) * 100:.1f}%.",
        "",
        "## Strongest classes",
        "",
        *[f"- {name}: {value * 100:.2f}%" for name, value in best_classes],
        "",
        "## Weakest classes",
        "",
        *[f"- {name}: {value * 100:.2f}%" for name, value in weak_classes],
        "",
        "## Figures",
        "",
        "- `01_total_loss.png`: total training loss and learning-rate changes.",
        "- `02_loss_components.png`: four Faster R-CNN loss components.",
        "- `03_ap50_by_class.png`: per-class AP@0.50 ranking.",
        "- `04_experiment_summary.png`: 16:9 summary dashboard for slides.",
    ]
    (output / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    training = read_training_csv(args.training_csv)
    metrics, per_class = read_evaluation_json(args.metrics_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_total_loss(training, args.output_dir, args.dpi)
    save_component_losses(training, args.output_dir, args.dpi)
    save_ap_chart(metrics, per_class, args.output_dir, args.dpi)
    save_summary_dashboard(training, metrics, per_class, args.output_dir, args.dpi)
    save_markdown_summary(training, metrics, per_class, args.output_dir)

    print(f"Figures saved to: {args.output_dir.resolve()}")
    for name in (
        "01_total_loss.png",
        "02_loss_components.png",
        "03_ap50_by_class.png",
        "04_experiment_summary.png",
        "experiment_summary.md",
    ):
        print(f"  - {name}")

    if args.show:
        for name in (
            "01_total_loss.png",
            "02_loss_components.png",
            "03_ap50_by_class.png",
            "04_experiment_summary.png",
        ):
            image = plt.imread(args.output_dir / name)
            plt.figure(figsize=(12, 7))
            plt.imshow(image)
            plt.axis("off")
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
