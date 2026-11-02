# SPDX-License-Identifier: Apache-2.0
"""Dense DMD sampling for MLXWan22DiT (Wan2.2 per-token timestep).

Matches the FastVideo pipeline's warped DMD schedule (``warp_denoising_step=True``,
``dmd_denoising_steps=[1000,757,522]``, ``flow_shift=5.0`` for TI2V-5B) rather
than treating raw step indices as continuous timesteps (a bug in early demos).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sequence

import numpy as np

from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step, pred_noise_to_pred_video

if TYPE_CHECKING:
    import mlx.core as mx

    from fastvideo.mlx_runtime.wan22 import MLXWan22DiT


def build_wan22_dmd_schedule(
    dmd_denoising_steps: Sequence[int] | None = None,
    *,
    flow_shift: float = 5.0,
    warp_denoising_step: bool = True,
) -> tuple[MLXDMDSchedule, list[float]]:
    """
    Build the flow-matching schedule and continuous timesteps used for Wan2.2 DMD sampling.

    Parameters:
        dmd_denoising_steps (Sequence[int] | None): Denoising step values to use; defaults to 1000, 757, and 522.
        flow_shift (float): Flow-matching shift applied when constructing the schedule.
        warp_denoising_step (bool): Whether to convert denoising steps to scheduler-warped continuous timesteps.

    Returns:
        tuple[MLXDMDSchedule, list[float]]: The DMD schedule and corresponding continuous timesteps.
    """
    import torch

    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    steps = list(dmd_denoising_steps or [1000, 757, 522])
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    schedule = MLXDMDSchedule.from_torch_scheduler(scheduler)
    step_idx = torch.tensor(steps, dtype=torch.long)
    if warp_denoising_step:
        warped = torch.cat((scheduler.timesteps.cpu(), torch.tensor([0.0], dtype=torch.float32)))
        timesteps = [float(t) for t in warped[1000 - step_idx]]
    else:
        timesteps = [float(s) for s in steps]
    return schedule, timesteps


def sample_wan22_dmd(
    model: MLXWan22DiT,
    encoder_hidden_states: mx.array,
    noise_latents: mx.array,
    freqs_cis: tuple,
    *,
    dmd_denoising_steps: Sequence[int] | None = None,
    flow_shift: float = 5.0,
    warp_denoising_step: bool = True,
    seed: int = 0,
    first_frame: mx.array | None = None,
    keyframes: Sequence[tuple[int, mx.array]] | None = None,
) -> mx.array:
    """
    Generate clean video latents from noisy latents using iterative DMD denoising.

    Parameters:
        noise_latents (mx.array): Initial noisy video latents.
        freqs_cis (tuple): Rotary positional frequency tensors used by the model.
        dmd_denoising_steps (Sequence[int] | None): DMD denoising steps, or the default schedule when omitted.
        flow_shift (float): Flow-matching schedule shift.
        warp_denoising_step (bool): Whether to warp the denoising timesteps.
        seed (int): Seed for reproducible intermediate re-noising.
        first_frame (mx.array | None): Optional clean, normalized first-frame latent
            ``[B, C, 1, H, W]`` for Wan2.2 TI2V image-to-video. When given, latent frame 0 is
            pinned to it: its tokens run at timestep 0 and it is restored after every step,
            mirroring ``WanDenoisingStage`` (``prepare_denoising`` / ``prepare_timestep`` /
            ``finish_step``) in the PyTorch pipeline.
        keyframes (Sequence[tuple[int, mx.array]] | None): EXPERIMENTAL extra pinned frames as
            ``(latent_frame_index, latent [B, C, 1, H, W])``; same pinning as ``first_frame`` at any
            latent position (e.g. the last frame for first+last-frame video). Indices must be unique
            and within ``[0, frames)``; ``first_frame`` counts as index 0.

    Returns:
        mx.array: Denoised video latents.
    """
    import mlx.core as mx

    schedule, timesteps = build_wan22_dmd_schedule(dmd_denoising_steps,
                                                   flow_shift=flow_shift,
                                                   warp_denoising_step=warp_denoising_step)
    # NumPy RNG so re-noise is bit-reproducible across MLX / torch A/B dumps.
    renoise_rng = np.random.default_rng(seed)
    latents = noise_latents
    batch, _c, frames, height, width = latents.shape
    pt, ph, pw = model.patch_size
    tokens = (frames // pt) * (height // ph) * (width // pw)

    # --- TI2V first-frame conditioning -------------------------------------------------
    pinned: dict[int, mx.array] = {}
    if first_frame is not None:
        pinned[0] = first_frame
    for idx, lat in (keyframes or ()):
        if idx in pinned:
            raise ValueError(f"duplicate pinned latent frame index {idx}")
        pinned[int(idx)] = lat
    keep = None            # [1,1,T,1,1]: 0 on pinned frames, 1 elsewhere
    pinned_tok = None      # [1, tokens] bool: token belongs to a pinned frame (runs at t = 0)
    ff_full = None         # pinned latents placed at their frame index, zeros elsewhere
    if pinned:
        if pt != 1:
            raise ValueError("keyframe conditioning assumes temporal patch size 1")
        for idx, lat in pinned.items():
            if not 0 <= idx < frames:
                raise ValueError(f"pinned latent frame index {idx} outside [0, {frames})")
            if lat.shape[2] != 1 or lat.shape[:2] != latents.shape[:2] or lat.shape[3:] != latents.shape[3:]:
                raise ValueError(f"pinned latent at {idx} has shape {tuple(lat.shape)}, expected "
                                 f"[{batch}, {latents.shape[1]}, 1, {height}, {width}]")
        tokens_per_frame = (height // ph) * (width // pw)
        zero = mx.zeros((batch, latents.shape[1], 1, height, width), dtype=mx.float32)
        ff_full = mx.concatenate([pinned[i].astype(mx.float32) if i in pinned else zero for i in range(frames)],
                                 axis=2)
        keep_np = np.ones((1, 1, frames, 1, 1), dtype=np.float32)
        for i in pinned:
            keep_np[:, :, i] = 0.0
        keep = mx.array(keep_np)
        pinned_tok = mx.array(np.repeat(keep_np.reshape(frames) == 0.0, tokens_per_frame)[None, :])
        latents = ((1.0 - keep) * ff_full + keep * latents.astype(mx.float32)).astype(noise_latents.dtype)

    def _pin(x):
        if keep is None:
            return x
        return ((1.0 - keep) * ff_full + keep * x.astype(mx.float32)).astype(x.dtype)

    last = len(timesteps) - 1
    for i, t in enumerate(timesteps):
        ts = mx.full((batch, tokens), float(t), dtype=mx.float32)
        if pinned_tok is not None:
            ts = mx.where(pinned_tok, mx.zeros_like(ts), ts)
        pred = model(latents.astype(mx.float16), encoder_hidden_states, ts, freqs_cis)
        ni = latents.astype(mx.float32)
        pn = pred.astype(mx.float32)
        if i < last:
            renoise = mx.array(renoise_rng.standard_normal(tuple(latents.shape)).astype(np.float32))
            latents = dmd_step(
                latents=ni,
                noise_input_latent=ni,
                pred_noise=pn,
                schedule=schedule,
                timestep=float(t),
                next_timestep=float(timesteps[i + 1]),
                noise=renoise,
            ).astype(latents.dtype)
        else:
            latents = pred_noise_to_pred_video(pn, ni, schedule.sigma_for(float(t))).astype(latents.dtype)
        latents = _pin(latents)
        mx.eval(latents)
    return latents
