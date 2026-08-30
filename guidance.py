from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _rgb_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).to(
        device=device, dtype=torch.float32)
    if value.max().item() > 1.5:
        value = value / 255.0
    return value.permute(2, 0, 1).unsqueeze(0).clamp_(0, 1)


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor((0.2126, 0.7152, 0.0722), device=rgb.device,
                           dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _horn_schunck(current: torch.Tensor, previous: torch.Tensor,
                  iterations: int, smoothness: float) -> torch.Tensor:
    """Estimate current-to-previous flow as source-pixel vectors."""
    kernel_x = torch.tensor(((-1, 1), (-1, 1)), device=current.device,
                            dtype=current.dtype).view(1, 1, 2, 2) * 0.25
    kernel_y = torch.tensor(((-1, -1), (1, 1)), device=current.device,
                            dtype=current.dtype).view(1, 1, 2, 2) * 0.25
    kernel_t = torch.ones((1, 1, 2, 2), device=current.device,
                          dtype=current.dtype) * 0.25
    pair = (current + previous) * 0.5
    ix = F.conv2d(pair, kernel_x, padding="same")
    iy = F.conv2d(pair, kernel_y, padding="same")
    it = F.conv2d(previous - current, kernel_t, padding="same")
    average = torch.tensor(((1 / 12, 1 / 6, 1 / 12),
                            (1 / 6, 0, 1 / 6),
                            (1 / 12, 1 / 6, 1 / 12)), device=current.device,
                           dtype=current.dtype).view(1, 1, 3, 3)
    u = torch.zeros_like(current)
    v = torch.zeros_like(current)
    denominator = smoothness * smoothness + ix * ix + iy * iy
    for _ in range(max(1, int(iterations))):
        u_avg = F.conv2d(u, average, padding=1)
        v_avg = F.conv2d(v, average, padding=1)
        correction = (ix * u_avg + iy * v_avg + it) / denominator.clamp_min(1e-6)
        u = u_avg - ix * correction
        v = v_avg - iy * correction
    return torch.cat((u, v), dim=1)


@torch.inference_mode()
def estimate_color_guidance(previous_rgb: np.ndarray | None,
                            current_rgb: np.ndarray,
                            motion_strength: float = 1.0,
                            depth_strength: float = 1.0,
                            iterations: int = 12,
                            analysis_max_side: int = 960,
                            gpu_index: int = 0):
    """Estimate approximate depth/motion/reactivity from RGB only.

    This is deliberately labelled an estimate: ordinary RGB does not contain
    engine-native geometry, camera matrices, projection jitter or motion data.
    """
    device = torch.device(
        f"cuda:{max(0, int(gpu_index))}" if torch.cuda.is_available() else "cpu")
    current = _rgb_tensor(current_rgb, device)
    height, width = current_rgb.shape[:2]
    scale = min(1.0, float(max(64, analysis_max_side)) / max(height, width))
    analysis_size = (max(16, int(round(height * scale))),
                     max(16, int(round(width * scale))))
    current_small = F.interpolate(current, size=analysis_size, mode="bilinear",
                                  align_corners=False)
    current_gray = _gray(current_small)

    # A lightweight color/contrast heuristic, not metric or geometric depth.
    local = F.avg_pool2d(current_gray, 15, stride=1, padding=7)
    contrast = (current_gray - local).abs()
    saturation = current_small.amax(1, keepdim=True) - current_small.amin(1, keepdim=True)
    depth = (0.62 * (1.0 - local) + 0.23 * contrast + 0.15 * saturation)
    depth = depth / depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    depth = (depth * float(depth_strength)).clamp(0, 1)

    if previous_rgb is None:
        flow = torch.zeros((1, 2, *analysis_size), device=device)
        reactive = torch.zeros_like(current_gray)
        exposure = 1.0
    else:
        previous = _rgb_tensor(previous_rgb, device)
        previous_small = F.interpolate(previous, size=analysis_size, mode="bilinear",
                                       align_corners=False)
        previous_gray = _gray(previous_small)
        flow = _horn_schunck(current_gray, previous_gray, iterations, 0.12)
        flow *= float(motion_strength)
        reactive = (current_small - previous_small).abs().amax(1, keepdim=True)
        reactive = (reactive * 4.0).clamp(0, 1)
        exposure = float((previous_gray.mean() / current_gray.mean().clamp_min(1e-5))
                         .clamp(0.25, 4.0).item())

    depth = F.interpolate(depth, size=(height, width), mode="bilinear",
                          align_corners=False)
    reactive = F.interpolate(reactive, size=(height, width), mode="bilinear",
                             align_corners=False)
    flow = F.interpolate(flow, size=(height, width), mode="bilinear",
                         align_corners=False)
    flow[:, 0] *= width / analysis_size[1]
    flow[:, 1] *= height / analysis_size[0]

    depth_np = depth[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
    motion_np = flow[0].permute(1, 2, 0).float().cpu().numpy().astype(np.float32, copy=False)
    reactive_np = reactive[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
    return depth_np, motion_np, reactive_np, exposure, device.type
