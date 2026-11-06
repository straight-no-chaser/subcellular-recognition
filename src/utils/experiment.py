import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import MicroscopyPatchDataset
from src.data.manifests import assign_group_splits, create_binary_labels, filter_metadata, load_metadata


def prepare_base_dataframe(data_config):
    df = load_metadata(data_config["manifest_paths"])
    split_config = data_config.get("split", {}) or {}

    existing_splits = pd.Series(dtype=str)
    if "split" in df.columns:
        existing_splits = df["split"].astype(str).replace("", pd.NA).dropna()

    needs_assignment = (
        split_config
        and (
            split_config.get("force", False)
            or not split_config.get("use_predefined", True)
            or existing_splits.empty
        )
    )

    if needs_assignment:
        df = assign_group_splits(
            df,
            seed=int(split_config.get("seed", 42)),
            train_frac=float(split_config.get("train_frac", 0.7)),
            val_frac=float(split_config.get("val_frac", 0.15)),
            test_frac=float(split_config.get("test_frac", 0.15)),
            group_col=split_config.get("group_col", None),
            force=True,
        )

    return df.reset_index(drop=True)


def prepare_task_dataframe(data_config, target_organelle=None, split=None, source=None):
    df = prepare_base_dataframe(data_config)
    df = filter_metadata(
        df,
        source=source if source is not None else data_config.get("source"),
        split=split,
        include_quality_flags=data_config.get("include_quality_flags"),
        exclude_quality_flags=data_config.get("exclude_quality_flags"),
    )

    target_organelle = target_organelle or data_config.get("target_organelle")
    if target_organelle:
        df = create_binary_labels(df, target_organelle)
    return df.reset_index(drop=True)


def make_dataset(df, config, augment=False, input_channels=None):
    data_config = config.get("data", config)
    model_config = config.get("model", {})
    input_channels = input_channels or data_config.get("input_channels") or model_config.get("in_channels", 1)

    return MicroscopyPatchDataset(
        df,
        image_size=int(data_config.get("image_size", 128)),
        input_channels=int(input_channels),
        crop_mode=data_config.get("crop_mode", "center"),
        normalize_mode=data_config.get("normalize_mode", "percentile"),
        percentile_low=float(data_config.get("percentile_low", 1.0)),
        percentile_high=float(data_config.get("percentile_high", 99.0)),
        augment=augment,
    )


def make_loader(dataset, loader_config, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=int(loader_config.get("batch_size", 32)),
        num_workers=int(loader_config.get("num_workers", 0)),
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
