from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


plt.rcParams["figure.dpi"] = 140


def plot_training_history(history_df, out_path):
    history_df = pd.DataFrame(history_df)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_auroc"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_auroc"], label="val")
    axes[1].set_title("AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_prob, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2)
    except Exception:
        ax.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_prob, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, lw=2)
    except Exception:
        ax.text(0.5, 0.5, "PR unavailable", ha="center", va="center")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, labels, out_path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(matrix, row_labels, col_labels, out_path, title="Transfer Heatmap", vmin=None, vmax=None):
    matrix = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_invalid(matrix)
    fig_width = max(5, 1.2 * len(col_labels))
    fig_height = max(4, 0.6 * len(row_labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_corruption_summary(df, metric, out_path):
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for corruption in sorted(df["corruption"].unique()):
        subset = df[df["corruption"] == corruption].sort_values("severity")
        ax.plot(subset["severity"], subset[metric], marker="o", label=corruption)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Corruption Sweep: {metric.upper()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_montage(images, titles, out_path, cols=5):
    n_images = max(len(images), 1)
    rows = ceil(n_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    if not images:
        for ax in axes.flat:
            ax.axis("off")
        axes.flat[0].text(0.5, 0.5, "No examples", ha="center", va="center")
    else:
        for ax, image, title in zip(axes.flat, images, titles):
            if image.ndim == 2:
                ax.imshow(image, cmap="gray")
            else:
                ax.imshow(image)
            ax.set_title(title[:48], fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(images) :]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
