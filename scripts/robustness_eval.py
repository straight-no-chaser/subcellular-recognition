import argparse

import pandas as pd
import torch

from src.eval.corruptions import AVAILABLE_CORRUPTIONS
from src.eval.runner import load_model_from_checkpoint, run_inference
from src.utils.config import dump_config, load_config, merge_dicts
from src.utils.experiment import make_dataset, make_loader, prepare_task_dataframe
from src.utils.io import make_run_dir, save_json
from src.utils.plots import plot_corruption_summary
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(args.ckpt, device)
    config = merge_dicts(checkpoint.get("config", {}), config)

    experiment = config.get("experiment", {})
    run_dir = make_run_dir(experiment.get("output_root", "results"), experiment.get("name", "robustness"))
    dump_config(config, run_dir / "config.yaml")

    data_config = config.get("data", {})
    target_organelle = data_config.get("target_organelle")
    split = config.get("evaluation", {}).get("split", "test")

    eval_df = prepare_task_dataframe(data_config, target_organelle=target_organelle, split=split)
    if eval_df.empty:
        raise ValueError("No samples matched the robustness evaluation filters")

    dataset = make_dataset(eval_df, config, augment=False, input_channels=checkpoint.get("in_channels", 1))
    loader = make_loader(dataset, config.get("loader", {}), shuffle=False)

    evaluation = config.get("evaluation", {})
    corruptions = evaluation.get("corruptions", AVAILABLE_CORRUPTIONS)
    severities = evaluation.get("severities", [0, 1, 2, 3, 4, 5])
    threshold = float(evaluation.get("threshold", 0.5))
    metric = evaluation.get("metric", "auroc")

    rows = []
    for corruption in corruptions:
        for severity in severities:
            setting = None if int(severity) == 0 else {"name": corruption, "severity": severity}
            _, metrics = run_inference(
                model,
                loader,
                device=device,
                threshold=threshold,
                corruption=setting,
            )
            rows.append(
                {
                    "corruption": corruption,
                    "severity": int(severity),
                    **metrics,
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(run_dir / "corruption_summary.csv", index=False)
    save_json(rows, run_dir / "corruption_summary.json")
    plot_corruption_summary(summary_df, metric=metric, out_path=run_dir / "corruption_plot.png")

    print(f"run_dir={run_dir}")
    print(summary_df[["corruption", "severity", metric]])


if __name__ == "__main__":
    main()
