import argparse

import pandas as pd
import torch

from src.data.manifests import create_binary_labels, filter_metadata
from src.eval.runner import load_model_from_checkpoint, run_inference
from src.utils.config import dump_config, load_config, merge_dicts
from src.utils.experiment import make_dataset, make_loader, prepare_base_dataframe
from src.utils.io import make_run_dir, save_json
from src.utils.plots import plot_heatmap
from src.utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = config.get("experiment", {})
    run_dir = make_run_dir(experiment.get("output_root", "results"), experiment.get("name", "transfer"))
    dump_config(config, run_dir / "config.yaml")

    data_config = config.get("data", {})
    sweep_config = config.get("sweep", {})
    base_df = prepare_base_dataframe(data_config)
    split = sweep_config.get("split", "test")
    sources = sweep_config.get("sources", data_config.get("source", None))
    base_df = filter_metadata(
        base_df,
        split=split,
        source=sources,
        include_quality_flags=data_config.get("include_quality_flags"),
        exclude_quality_flags=data_config.get("exclude_quality_flags"),
    )

    if base_df.empty:
        raise ValueError("No samples matched the transfer sweep filters")

    rows = []
    column_keys = []
    multi_source = len(sources) > 1 if isinstance(sources, list) else False

    for run in sweep_config.get("runs", []):
        ckpt_path = run["ckpt"]
        train_organelle = run.get("train_organelle", "")
        model, checkpoint = load_model_from_checkpoint(ckpt_path, device)
        merged_config = merge_dicts(checkpoint.get("config", {}), config)
        input_channels = checkpoint.get("in_channels", merged_config.get("model", {}).get("in_channels", 1))

        run_sources = run.get("sources", sources)
        run_sources = run_sources if isinstance(run_sources, list) else [run_sources]

        for source in run_sources:
            source_df = filter_metadata(base_df, source=[source] if source else None)
            for test_organelle in sweep_config.get("test_organelles", []):
                eval_df = create_binary_labels(source_df, test_organelle)
                dataset = make_dataset(eval_df, merged_config, augment=False, input_channels=input_channels)
                loader = make_loader(dataset, merged_config.get("loader", {}), shuffle=False)
                _, metrics = run_inference(
                    model,
                    loader,
                    device=device,
                    threshold=float(merged_config.get("evaluation", {}).get("threshold", 0.5)),
                )
                key = f"{test_organelle}@{source}" if multi_source else test_organelle
                column_keys.append(key)
                rows.append(
                    {
                        "train_organelle": train_organelle,
                        "test_organelle": test_organelle,
                        "source": source,
                        "column_key": key,
                        "checkpoint": ckpt_path,
                        **metrics,
                    }
                )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(run_dir / "transfer_summary.csv", index=False)
    save_json(rows, run_dir / "transfer_summary.json")

    metric = sweep_config.get("metric", "auroc")
    row_labels = list(dict.fromkeys(summary_df["train_organelle"].tolist()))
    col_labels = list(dict.fromkeys(column_keys))
    heatmap_df = summary_df.pivot_table(index="train_organelle", columns="column_key", values=metric, aggfunc="mean")
    heatmap_df = heatmap_df.reindex(index=row_labels, columns=col_labels)
    plot_heatmap(
        heatmap_df.values,
        row_labels=row_labels,
        col_labels=col_labels,
        out_path=run_dir / "transfer_heatmap.png",
        title=f"Transfer Heatmap ({metric.upper()})",
    )

    print(f"run_dir={run_dir}")
    print(summary_df[["train_organelle", "test_organelle", "source", metric]])


if __name__ == "__main__":
    main()
