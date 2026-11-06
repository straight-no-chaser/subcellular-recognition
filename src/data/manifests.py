from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DEFAULT_COLUMNS = [
    "image_path",
    "source",
    "cell_line",
    "organelle_label",
    "channels",
    "bit_depth",
    "split",
    "split_group",
    "acquisition_batch",
    "quality_flags",
]


def _ensure_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    return [str(value)]


def _normalize_text(value, default=""):
    if pd.isna(value):
        return default
    return str(value).strip()


def _resolve_image_path(image_path, manifest_dir):
    image_path = _normalize_text(image_path)
    if not image_path:
        return ""
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str((manifest_dir / path).resolve())


def load_metadata(manifest_paths):
    manifest_paths = _ensure_list(manifest_paths)
    if not manifest_paths:
        raise ValueError("manifest_paths is empty")

    frames = []
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path)
        for column in DEFAULT_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        df["manifest_path"] = str(manifest_path.resolve())
        df["image_path"] = [
            _resolve_image_path(path, manifest_path.parent) for path in df["image_path"]
        ]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    for column in DEFAULT_COLUMNS:
        df[column] = df[column].fillna("")
    df["source"] = df["source"].replace("", "unknown")
    return df


def _tokenize_flags(value):
    text = _normalize_text(value).lower()
    if not text:
        return set()
    return {token for token in re.split(r"[;,| ]+", text) if token}


def filter_metadata(
    df,
    source=None,
    split=None,
    organelles=None,
    split_groups=None,
    include_quality_flags=None,
    exclude_quality_flags=None,
):
    out = df.copy()

    source = _ensure_list(source)
    split = _ensure_list(split)
    organelles = _ensure_list(organelles)
    split_groups = _ensure_list(split_groups)
    include_quality_flags = _ensure_list(include_quality_flags)
    exclude_quality_flags = _ensure_list(exclude_quality_flags)

    if source:
        out = out[out["source"].astype(str).isin(source)]
    if split:
        out = out[out["split"].astype(str).isin(split)]
    if organelles:
        out = out[out["organelle_label"].astype(str).isin(organelles)]
    if split_groups:
        out = out[out["split_group"].astype(str).isin(split_groups)]
    if include_quality_flags:
        required = {x.lower() for x in include_quality_flags}
        mask = out["quality_flags"].map(lambda x: required.issubset(_tokenize_flags(x)))
        out = out[mask]
    if exclude_quality_flags:
        banned = {x.lower() for x in exclude_quality_flags}
        mask = out["quality_flags"].map(lambda x: not bool(_tokenize_flags(x) & banned))
        out = out[mask]

    return out.reset_index(drop=True)


def create_binary_labels(df, target_organelle):
    target_organelle = str(target_organelle).strip().lower()
    out = df.copy()
    labels = out["organelle_label"].astype(str).str.lower()
    out["label"] = (labels == target_organelle).astype(np.float32)
    out["target_organelle"] = target_organelle
    return out


def pick_group_column(df, preferred=None):
    preferred = preferred or ["split_group", "acquisition_batch"]
    for column in preferred:
        if column in df.columns:
            values = df[column].astype(str).replace("", np.nan).dropna()
            if values.nunique() > 1:
                return column
    return None


def assign_group_splits(
    df,
    seed=42,
    train_frac=0.7,
    val_frac=0.15,
    test_frac=0.15,
    group_col=None,
    force=False,
):
    out = df.copy()

    existing = set(out.get("split", pd.Series(dtype=str)).astype(str).str.lower().unique())
    if not force and {"train", "val", "test"} & existing:
        return out

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    group_col = group_col or pick_group_column(out)
    if group_col is None:
        out["__group__"] = [f"sample_{i}" for i in range(len(out))]
        group_col = "__group__"

    groups = out[group_col].astype(str).replace("", np.nan)
    groups = groups.fillna(pd.Series([f"sample_{i}" for i in range(len(out))], index=out.index))
    out[group_col] = groups

    n_groups = groups.nunique()
    if len(out) < 10 or n_groups < 3:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(out))
        n_train = int(len(out) * train_frac)
        n_val = int(len(out) * val_frac)
        split = np.array(["test"] * len(out), dtype=object)
        split[order[:n_train]] = "train"
        split[order[n_train : n_train + n_val]] = "val"
        out["split"] = split
        if "__group__" in out.columns:
            out = out.drop(columns="__group__")
        return out

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(splitter_test.split(out, groups=groups))
    train_val = out.iloc[train_val_idx].copy()
    test = out.iloc[test_idx].copy()

    val_ratio = val_frac / max(train_frac + val_frac, 1e-8)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(
        splitter_val.split(train_val, groups=train_val[group_col].astype(str))
    )

    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    out = pd.concat([train, val, test], ignore_index=True)
    if "__group__" in out.columns:
        out = out.drop(columns="__group__")
    return out
