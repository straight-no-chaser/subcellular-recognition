# Cross-organelle transfer in fluorescence microscopy

Organelle-specific binary classification assessing transferability, cross-dataset shift, robustness, and failure analysis on Human Protein Atlas (HPA) and Allen Cell Explorer style manifests, with optional BBBC or Cell Painting support through the same metadata layer.

## Train

```bash
python scripts/train.py --config configs/train_hpa_mito_resnet18.yaml
python scripts/train.py --config configs/train_hpa_golgi_efficientnet.yaml
```

## Evaluation

```bash
python scripts/eval.py --config configs/eval_allen_mito.yaml --ckpt results/train_hpa_mito_resnet18_YYYYMMDD_HHMMSS/best.pt
python scripts/eval.py --config configs/eval_allen_mito.yaml --ckpt results/train_hpa_mito_resnet18_YYYYMMDD_HHMMSS/best.pt --target_organelle golgi
```

## Transfer Analysis Workflow

Consumes one or more trained checkpoints and evaluates each model across a panel of organelles, optionally across multiple data sources.

```bash
python scripts/transfer_sweep.py --config configs/transfer_hpa_panel.yaml
```

## Robustness Workflow

Evaluation-time corruptions with perturbations:

- gaussian noise
- blur
- intensity shift
- channel dropout
- bleed-through-like channel mixing

```bash
python scripts/robustness_eval.py --config configs/eval_corruptions_hpa_mito.yaml --ckpt results/train_hpa_mito_resnet18_YYYYMMDD_HHMMSS/best.pt
```

## Failure Analysis

```bash
python scripts/failure_analysis.py --pred_csv results/eval_allen_mito_YYYYMMDD_HHMMSS/predictions.csv
```