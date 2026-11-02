# SPDX-License-Identifier: Apache-2.0
"""Image conditioning helpers for the Wan2.2 TI2V-5B MLX runtime (image-to-video / keyframes).

Mirrors the PyTorch TI2V path:

* ``InputValidationStage``      -> resize-to-cover + center-crop, scale to [-1, 1]
* ``WanFirstFrameEncodingStage`` -> ``vae.encode(img).mean`` then ``(z - mean) / std``

The result is a clean, *normalized* latent ``[1, 48, 1, H/16, W/16]`` -- the same space the DiT
works in and that TAEHV decodes -- ready for ``sample_wan22_dmd(first_frame=... / keyframes=...)``.

Keyframes after the first frame (EXPERIMENTAL): the causal Wan VAE encodes a lone image as a
"first frame"; a later latent frame stands for a chunk of 4 pixel frames.  For those positions the
image is encoded as a static 5-frame clip (image repeated) and latent frame 1 is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def load_and_fit_image(image_path: str | Path, height: int, width: int):
    """Load an image and fit it to ``width x height`` (scale to cover, then center-crop).

    Same geometry as FastVideo's Wan TI2V input handling, so aspect ratio is preserved and no
    stretching happens. Returns ``(PIL.Image, info_str)``.
    """
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    iw, ih = img.size
    scale = max(width / iw, height / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - width) // 2
    y1 = (img.height - height) // 2
    img = img.crop((x1, y1, x1 + width, y1 + height))
    assert img.size == (width, height)
    kept = 100.0 * min(1.0, (width / scale) * (height / scale) / (iw * ih))
    return img, f"{iw}x{ih} -> {width}x{height} (kept {kept:.0f}% of the image area)"


def encode_keyframe_latents(
    items: Sequence[tuple[str | Path, bool]],
    vae_dir: Path,
    *,
    height: int,
    width: int,
    device: str = "auto",
    dtype_name: str = "fp32",
) -> list[tuple[np.ndarray, str]]:
    """Encode images into normalized latents with the full Wan2.2 VAE (loaded once).

    ``items`` is a list of ``(image_path, is_first_frame)``.  ``is_first_frame=True`` encodes the
    image as a single frame (standard TI2V first-frame latent).  ``False`` encodes it as a static
    5-frame clip and returns latent frame 1 (a "mid/last" chunk latent).

    Returns a list of ``(latent [1, C, 1, height/16, width/16] float32, fit_info)``.
    """
    import torch
    from diffusers import AutoencoderKLWan

    from fastvideo.mlx_runtime.memory import cleanup_torch_mps
    from fastvideo.mlx_runtime.wan_vae import WanVAEConfigView

    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if (dtype_name == "fp16" and device != "cpu") else torch.float32

    cfg = WanVAEConfigView.from_vae_dir(vae_dir)
    vae = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype, local_files_only=True).to(device).eval()

    out: list[tuple[np.ndarray, str]] = []
    for image_path, is_first in items:
        img, info = load_and_fit_image(image_path, height, width)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # H W C in [0, 1]
        x = x.permute(2, 0, 1).mul(2.0).sub(1.0)                         # C H W in [-1, 1]
        x = x[None, :, None]                                             # B C 1 H W
        if not is_first:
            x = x.repeat(1, 1, 5, 1, 1)                                  # static 5-frame clip
        x = x.to(device=device, dtype=dtype)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode().float()                # == posterior mean
        if not is_first:
            z = z[:, :, 1:2]
        mean = torch.tensor(cfg.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(cfg.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        out.append((((z - mean) / std).cpu().numpy().astype(np.float32), info))

    del vae
    cleanup_torch_mps()
    return out


def encode_first_frame_latent(
    image_path: str | Path,
    vae_dir: Path,
    *,
    height: int,
    width: int,
    device: str = "auto",
    dtype_name: str = "fp32",
) -> tuple[np.ndarray, str]:
    """Encode one image into a normalized first-frame latent (single-image TI2V)."""
    return encode_keyframe_latents([(image_path, True)], vae_dir, height=height, width=width, device=device,
                                   dtype_name=dtype_name)[0]
