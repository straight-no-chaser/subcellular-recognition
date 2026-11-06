import pandas as pd
import torch

from src.eval.corruptions import apply_corruption
from src.models.backbones import build_model
from src.utils.metrics import compute_binary_metrics


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})

    model = build_model(
        checkpoint.get("arch", model_config.get("arch", "resnet18")),
        in_channels=int(checkpoint.get("in_channels", model_config.get("in_channels", 1))),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _extend_metadata(store, batch):
    for key, value in batch.items():
        if key in ["image", "label"]:
            continue
        if torch.is_tensor(value):
            store.setdefault(key, []).extend(value.detach().cpu().tolist())
        elif isinstance(value, (list, tuple)):
            store.setdefault(key, []).extend(list(value))
        else:
            store.setdefault(key, []).append(value)


@torch.no_grad()
def run_inference(model, loader, device, criterion=None, threshold=0.5, corruption=None):
    model.eval()

    losses = []
    logits_all = []
    probs_all = []
    labels_all = []
    metadata = {}

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).view(-1)

        if corruption:
            images = apply_corruption(
                images,
                name=corruption.get("name"),
                severity=corruption.get("severity", 0),
            )

        logits = model(images).view(-1)
        probs = torch.sigmoid(logits)

        if criterion is not None:
            losses.append(float(criterion(logits, labels).item()))

        logits_all.extend(logits.detach().cpu().numpy().tolist())
        probs_all.extend(probs.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        _extend_metadata(metadata, batch)

    if not labels_all:
        raise ValueError("No samples were available for evaluation")

    predictions = pd.DataFrame(metadata)
    predictions["label"] = [int(x) for x in labels_all]
    predictions["logit"] = logits_all
    predictions["prob"] = probs_all
    predictions["pred_label"] = (predictions["prob"] >= threshold).astype(int)

    error_type = []
    for label, pred in zip(predictions["label"], predictions["pred_label"]):
        if label == 1 and pred == 1:
            error_type.append("tp")
        elif label == 0 and pred == 1:
            error_type.append("fp")
        elif label == 1 and pred == 0:
            error_type.append("fn")
        else:
            error_type.append("tn")
    predictions["error_type"] = error_type

    metrics = compute_binary_metrics(predictions["label"], predictions["prob"], threshold=threshold)
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    return predictions, metrics
