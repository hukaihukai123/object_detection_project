# Faster R-CNN experiment summary

## Main results

| Metric | Result |
|---|---:|
| mAP@0.50 | 79.64% |
| mAP@0.50:0.95 | 50.27% |
| Precision@0.50 | 24.46% |
| Recall@0.50 | 91.98% |
| Inference speed | 16.42 images/s |
| Test images | 4952 |

## Training convergence

Total loss decreased from 0.3517 to 0.1697 over 15 epochs, a reduction of 51.7%.

## Strongest classes

- car: 90.22%
- person: 88.05%
- bicycle: 87.62%
- horse: 87.53%
- cat: 87.03%

## Weakest classes

- pottedplant: 51.16%
- chair: 66.12%
- boat: 67.69%
- bottle: 67.74%
- bird: 75.77%

## Figures

- `01_total_loss.png`: total training loss and learning-rate changes.
- `02_loss_components.png`: four Faster R-CNN loss components.
- `03_ap50_by_class.png`: per-class AP@0.50 ranking.
- `04_experiment_summary.png`: 16:9 summary dashboard for slides.