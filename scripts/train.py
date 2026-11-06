import argparse

import torch

from src.models.backbones import build_model
from src.train.engine import train_model
from src.utils.config import dump_config, load_config
from src.utils.experiment import make_dataset, make_loader, prepare_task_dataframe
from src.utils.io import make_run_dir
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
    run_dir = make_run_dir(experiment.get("output_root", "results"), experiment.get("name", "train"))
    dump_config(config, run_dir / "config.yaml")

    target_organelle = config.get("data", {}).get("target_organelle")
    labeled_df = prepare_task_dataframe(config.get("data", {}), target_organelle=target_organelle, split=None)

    train_df = labeled_df[labeled_df["split"] == "train"].reset_index(drop=True)
    val_df = labeled_df[labeled_df["split"] == "val"].reset_index(drop=True)
    test_df = labeled_df[labeled_df["split"] == "test"].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Training and validation splits must both contain samples")

    labeled_df.to_csv(run_dir / "manifest.csv", index=False)
    train_df.to_csv(run_dir / "train_manifest.csv", index=False)
    val_df.to_csv(run_dir / "val_manifest.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(run_dir / "test_manifest.csv", index=False)

    train_dataset = make_dataset(train_df, config, augment=True)
    val_dataset = make_dataset(val_df, config, augment=False)
    train_loader = make_loader(train_dataset, config.get("loader", {}), shuffle=True)
    val_loader = make_loader(val_dataset, config.get("loader", {}), shuffle=False)

    model = build_model(
        config.get("model", {}).get("arch", "resnet18"),
        in_channels=int(config.get("model", {}).get("in_channels", 1)),
        pretrained=bool(config.get("model", {}).get("pretrained", False)),
    ).to(device)

    history = train_model(model, train_loader, val_loader, train_df, run_dir, config, device)
    history.to_csv(run_dir / "history.csv", index=False)

    print(f"run_dir={run_dir}")
    print(f"train_samples={len(train_df)} val_samples={len(val_df)}")


if __name__ == "__main__":
    main()
