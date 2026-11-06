import math

import torch
import torch.nn.functional as F


AVAILABLE_CORRUPTIONS = [
    "gaussian_noise",
    "blur",
    "intensity_shift",
    "channel_dropout",
    "channel_mix",
]


def _gaussian_kernel(kernel_size, sigma, channels, device):
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel_2d


def gaussian_blur(images, severity):
    kernel_size = int(2 * severity + 1)
    sigma = 0.5 + 0.6 * severity
    kernel = _gaussian_kernel(kernel_size, sigma, images.shape[1], images.device)
    padding = kernel_size // 2
    return F.conv2d(images, kernel, padding=padding, groups=images.shape[1])


def apply_corruption(images, name=None, severity=0):
    if not name or int(severity) <= 0:
        return images

    name = str(name).lower()
    severity = float(severity)
    out = images.clone()

    if name == "gaussian_noise":
        scale = 0.03 * severity
        out = out + torch.randn_like(out) * scale

    elif name == "blur":
        out = gaussian_blur(out, severity)

    elif name == "intensity_shift":
        shift = 0.06 * severity
        scale = max(0.0, 1.0 - 0.04 * severity)
        out = out * scale + shift

    elif name == "channel_dropout":
        if out.shape[1] == 1:
            out = out * max(0.0, 1.0 - 0.18 * severity)
        else:
            for index in range(out.shape[0]):
                channel = index % out.shape[1]
                out[index, channel] = 0.0

    elif name in ["channel_mix", "bleedthrough", "bleed_through"]:
        if out.shape[1] == 1:
            blurred = gaussian_blur(out, max(severity, 1.0))
            out = 0.8 * out + 0.2 * blurred
        else:
            rolled = torch.roll(out, shifts=1, dims=1)
            alpha = min(0.12 * severity, 0.45)
            out = (1.0 - alpha) * out + alpha * rolled

    else:
        raise ValueError(f"Unknown corruption: {name}")

    return out.clamp(0.0, 1.0)
