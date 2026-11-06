from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.eval.runner import run_inference
from src.utils.io import save_json
from src.utils.plots import plot_training_history
from src.utils.metrics import compute_binary_metrics


def create_optimizer(model, train_config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 3e-4)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )


def create_scheduler(optimizer, train_config):
    scheduler_config = train_config.get("scheduler", {}) or {}
    name = str(scheduler_config.get("name", "cosine")).lower()

    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(train_config.get("epochs", 20)),
                eta_min=float(scheduler_config.get("min_lr", 1e-6)),
            ),
            name,
        )

    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(scheduler_config.get("factor", 0.5)),
                patience=int(scheduler_config.get("patience", 2)),
                min_lr=float(scheduler_config.get("min_lr", 1e-6)),
            ),
            name,
        )

    return None, None


def resolve_pos_weight(train_df, train_config, device):
    pos_weight = train_config.get("pos_weight", None)
    if pos_weight in [None, "", "none", False]:
        return None

    if str(pos_weight).lower() == "auto":
        positives = max(float(train_df["label"].sum()), 1.0)
        negatives = max(float(len(train_df) - train_df["label"].sum()), 1.0)
        pos_weight = negatives / positives

    return torch.tensor([float(pos_weight)], device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    probs = []
    labels = []

    for batch in loader:
        images = batch["image"].to(device)
        target = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).view(-1)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        labels.extend(target.detach().cpu().numpy().tolist())

    metrics = compute_binary_metrics(labels, probs)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def checkpoint_state(model, optimizer, scheduler, epoch, best_metric, config):
    return {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "arch": config.get("model", {}).get("arch", ""),
        "in_channels": config.get("model", {}).get("in_channels", 1),
        "train_organelle": config.get("data", {}).get("target_organelle", ""),
    }


def load_resume(model, optimizer, scheduler, resume_path, device):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric = float(checkpoint.get("best_metric", -np.inf))
    return start_epoch, best_metric


def train_model(model, train_loader, val_loader, train_df, run_dir, config, device):
    train_config = config.get("train", {})
    run_dir = Path(run_dir)

    pos_weight = resolve_pos_weight(train_df, train_config, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = create_optimizer(model, train_config)
    scheduler, scheduler_name = create_scheduler(optimizer, train_config)

    start_epoch = 0
    best_metric = -np.inf
    history_path = run_dir / "history.csv"
    history = []

    if history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")

    resume_path = train_config.get("resume_ckpt", None)
    if resume_path:
        start_epoch, best_metric = load_resume(model, optimizer, scheduler, resume_path, device)

    epochs = int(train_config.get("epochs", 20))
    threshold = float(train_config.get("threshold", 0.5))

    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_predictions, val_metrics = run_inference(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            threshold=threshold,
        )

        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics.get("loss", float("nan")),
            "val_auroc": val_metrics.get("auroc", float("nan")),
            "val_auprc": val_metrics.get("auprc", float("nan")),
            "val_accuracy": val_metrics.get("accuracy", float("nan")),
            "val_f1": val_metrics.get("f1", float("nan")),
        }
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(history_path, index=False)
        plot_training_history(history_df, run_dir / "training_curves.png")

        current_metric = val_metrics.get("auroc", float("nan"))
        improved = not np.isnan(current_metric) and current_metric > best_metric

        if improved:
            best_metric = current_metric
            state = checkpoint_state(model, optimizer, scheduler, epoch, best_metric, config)
            torch.save(state, run_dir / "best.pt")
            val_predictions.to_csv(run_dir / "best_val_predictions.csv", index=False)
            save_json(val_metrics, run_dir / "best_val_metrics.json")

        state = checkpoint_state(model, optimizer, scheduler, epoch, best_metric, config)
        torch.save(state, run_dir / "last.pt")

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(current_metric if not np.isnan(current_metric) else row["val_loss"])
            else:
                scheduler.step()

    final_history = pd.read_csv(history_path)
    plot_training_history(final_history, run_dir / "training_curves.png")
    return final_history
