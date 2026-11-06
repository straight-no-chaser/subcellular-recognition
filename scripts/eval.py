import argparse
import numpy as np
import torch

from src.eval.runner import load_model_from_checkpoint, run_inference
from src.utils.config import dump_config, load_config, merge_dicts
from src.utils.experiment import make_dataset, make_loader, prepare_task_dataframe
from src.utils.io import make_run_dir, save_json
from src.utils.plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--target_organelle", default=None)
    parser.add_argument("--source", nargs="+", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--severity", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    eval_config = load_config(args.config)
    seed_everything(int(eval_config.get("experiment", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(args.ckpt, device)
    ckpt_config = checkpoint.get("config", {})
    config = merge_dicts(ckpt_config, eval_config)

    experiment = config.get("experiment", {})
    run_dir = make_run_dir(experiment.get("output_root", "results"), experiment.get("name", "eval"))
    dump_config(config, run_dir / "config.yaml")

    data_config = config.get("data", {})
    target_organelle = args.target_organelle or data_config.get("target_organelle")
    split = args.split or config.get("evaluation", {}).get("split", "test")
    source = args.source if args.source is not None else data_config.get("source")

    data_config = merge_dicts(
        data_config,
        {"target_organelle": target_organelle, "source": source},
    )

    eval_df = prepare_task_dataframe(data_config, target_organelle=target_organelle, split=split, source=source)
    if eval_df.empty:
        raise ValueError("No samples matched the evaluation filters")

    eval_df.to_csv(run_dir / "evaluation_manifest.csv", index=False)
    dataset = make_dataset(eval_df, config, augment=False, input_channels=checkpoint.get("in_channels", 1))
    loader = make_loader(dataset, config.get("loader", {}), shuffle=False)

    corruption = config.get("evaluation", {}).get("corruption", {}) or {}
    if args.corruption is not None:
        corruption["name"] = args.corruption
    if args.severity is not None:
        corruption["severity"] = args.severity
    if not corruption.get("name"):
        corruption = None

    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    predictions, metrics = run_inference(
        model,
        loader,
        device=device,
        threshold=threshold,
        corruption=corruption,
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)

    summary = {
        "checkpoint": args.ckpt,
        "train_organelle": checkpoint.get("train_organelle", ""),
        "target_organelle": target_organelle,
        "source": source,
        "split": split,
        "corruption": corruption or {"name": "clean", "severity": 0},
        "metrics": metrics,
    }
    save_json(summary, run_dir / "summary.json")

    plot_roc_curve(predictions["label"], predictions["prob"], run_dir / "roc_curve.png")
    plot_pr_curve(predictions["label"], predictions["prob"], run_dir / "pr_curve.png")
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plot_confusion_matrix(cm, ["negative", "positive"], run_dir / "confusion_matrix.png")

    print(f"run_dir={run_dir}")
    print(summary)


if __name__ == "__main__":
    main()
