from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_image(path):
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in [".tif", ".tiff"]:
        array = tifffile.imread(path)
    else:
        array = np.array(Image.open(path))

    array = np.asarray(array).squeeze()
    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3:
        if array.shape[0] in [1, 2, 3, 4] and array.shape[-1] not in [1, 2, 3, 4]:
            array = np.moveaxis(array, 0, -1)
    else:
        raise ValueError(f"Unsupported image shape for {path}: {array.shape}")

    return array.astype(np.float32)


def normalize_image(array, mode="percentile", percentile_low=1.0, percentile_high=99.0, bit_depth=None):
    array = array.astype(np.float32)
    if mode == "bit_depth" and str(bit_depth).strip() not in ["", "nan", "None"]:
        scale = max((2 ** int(float(bit_depth))) - 1, 1)
        array = array / float(scale)
    else:
        lo = np.percentile(array, percentile_low, axis=(0, 1), keepdims=True)
        hi = np.percentile(array, percentile_high, axis=(0, 1), keepdims=True)
        array = (array - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(array, 0.0, 1.0)


def adapt_channels(array, input_channels):
    channels = array.shape[-1]
    if channels == input_channels:
        return array
    if input_channels == 1:
        return array.mean(axis=-1, keepdims=True)
    if channels == 1:
        return np.repeat(array, input_channels, axis=-1)
    if channels > input_channels:
        return array[..., :input_channels]
    pad = np.repeat(array[..., -1:], input_channels - channels, axis=-1)
    return np.concatenate([array, pad], axis=-1)


def center_crop_tensor(tensor):
    _, height, width = tensor.shape
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return tensor[:, top : top + size, left : left + size]


def prepare_image_tensor(
    path,
    image_size=128,
    input_channels=1,
    crop_mode="center",
    normalize_mode="percentile",
    percentile_low=1.0,
    percentile_high=99.0,
    bit_depth=None,
):
    array = read_image(path)
    array = normalize_image(
        array,
        mode=normalize_mode,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        bit_depth=bit_depth,
    )
    array = adapt_channels(array, input_channels)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    if crop_mode == "center":
        tensor = center_crop_tensor(tensor)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return tensor.clamp(0.0, 1.0)


def prepare_display_image(path):
    array = read_image(path)
    array = normalize_image(array)
    array = adapt_channels(array, min(max(array.shape[-1], 1), 3))
    if array.shape[-1] == 1:
        return array[..., 0]
    if array.shape[-1] == 2:
        third = array[..., :1]
        return np.concatenate([array, third], axis=-1)
    return array[..., :3]


class MicroscopyPatchDataset(Dataset):
    def __init__(
        self,
        df,
        image_size=128,
        input_channels=1,
        crop_mode="center",
        normalize_mode="percentile",
        percentile_low=1.0,
        percentile_high=99.0,
        augment=False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.image_size = image_size
        self.input_channels = input_channels
        self.crop_mode = crop_mode
        self.normalize_mode = normalize_mode
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        tensor = prepare_image_tensor(
            row["image_path"],
            image_size=self.image_size,
            input_channels=self.input_channels,
            crop_mode=self.crop_mode,
            normalize_mode=self.normalize_mode,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            bit_depth=row.get("bit_depth", None),
        )

        if self.augment:
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
            if torch.rand(1).item() < 0.5:
                tensor = torch.flip(tensor, dims=[1])

        sample = {
            "image": tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "image_path": str(row.get("image_path", "")),
            "source": str(row.get("source", "")),
            "cell_line": str(row.get("cell_line", "")),
            "organelle_label": str(row.get("organelle_label", "")),
            "target_organelle": str(row.get("target_organelle", "")),
            "channels": str(row.get("channels", "")),
            "bit_depth": str(row.get("bit_depth", "")),
            "split": str(row.get("split", "")),
            "split_group": str(row.get("split_group", "")),
            "acquisition_batch": str(row.get("acquisition_batch", "")),
            "quality_flags": str(row.get("quality_flags", "")),
        }
        return sample
