from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageSequence

from .guidance import estimate_color_guidance
from .nodes import (
    Bridge,
    Settings,
    _comparison_rgba,
    _depth_frame,
    _event_for,
    _motion_frame,
    _preview,
    _remove_event,
    _rgba8,
    gpu_index,
    gpu_input,
    runtime_input,
)


def _style(value) -> int:
    return int(str(value).split()[0])


def _guidance(value) -> int:
    return int(str(value).split()[0])


def _settings(style, intensity, tone, structure, automatic_mask,
              skin_structure=1.0, ui_correction=False, depth_inverted=1,
              motion_scale_x=1.0, motion_scale_y=1.0) -> Settings:
    return Settings(_style(style), intensity, tone, structure, int(automatic_mask), 0,
                    skin_structure, int(ui_correction), int(depth_inverted),
                    motion_scale_x, motion_scale_y, 1.0)


def _tensor_rgb(rgb: np.ndarray, device=None) -> torch.Tensor:
    result = torch.from_numpy(np.ascontiguousarray(rgb[..., :3])).float() / 255.0
    return result.to(device) if device is not None else result


def _check_interrupt() -> None:
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _output_directory() -> Path:
    try:
        import folder_paths
        result = Path(folder_paths.get_output_directory()) / "dlssnr"
    except Exception:
        result = Path.cwd() / "output" / "dlssnr"
    result.mkdir(parents=True, exist_ok=True)
    return result


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


class DLSSNRColorGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gpu_device": gpu_input(),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "flow_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
                "analysis_max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 64}),
            },
            "optional": {"previous_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "DLSSNR_MOTION", "MASK", "FLOAT", "STRING")
    RETURN_NAMES = ("estimated_depth", "motion_vectors", "reactive_mask",
                    "exposure_ratio", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Guidance"

    def run(self, image, gpu_device, motion_strength, depth_strength, flow_iterations,
            analysis_max_side, previous_image=None):
        frames = image.detach().float().cpu().clamp(0, 1).numpy()[..., :3]
        depths = []
        motions = []
        masks = []
        exposures = []
        previous = None
        previous_frames = (None if previous_image is None else
                           previous_image.detach().float().cpu().clamp(0, 1).numpy()[..., :3])
        device_name = "cpu"
        selected_gpu = gpu_index(gpu_device)
        for index, frame in enumerate(frames):
            current = np.ascontiguousarray(frame * 255.0 + 0.5, dtype=np.uint8)
            if previous_frames is not None:
                source = previous_frames[min(index, len(previous_frames) - 1)]
                previous = np.ascontiguousarray(source * 255.0 + 0.5, dtype=np.uint8)
            depth, motion, mask, exposure, device_name = estimate_color_guidance(
                previous, current, motion_strength, depth_strength,
                flow_iterations, analysis_max_side, selected_gpu)
            depths.append(torch.from_numpy(np.repeat(depth[..., None], 3, axis=-1).copy()))
            motions.append(torch.from_numpy(motion.copy()))
            masks.append(torch.from_numpy(mask.copy()))
            exposures.append(exposure)
            previous = current
        status = (f"RGB估算完成：{len(frames)} 帧，计算设备={device_name.upper()}。"
                  "深度/光流是颜色推断，不是游戏引擎原生数据。")
        return (torch.stack(depths), torch.stack(motions), torch.stack(masks),
                float(np.mean(exposures)), status)


class DLSSNRImageDirect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度"],),
                "depth_convention": (["1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "estimate_missing_from_color": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "motion_vectors": ("DLSSNR_MOTION",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "comparison_image", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, image, dll_path, gpu_device, automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, ui_correction, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            estimate_missing_from_color, unique_id=None, depth=None,
            motion_vectors=None):
        batch, height, width, _ = image.shape
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y)
        processed_results = []
        comparison_results = []
        previous = None
        last_comparison = None
        estimator_device = "unused"
        selected_gpu = gpu_index(gpu_device)
        started = time.perf_counter()
        with Bridge(dll_path, width, height, selected_gpu) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = _depth_frame(depth, index, 1, height, width) if want_depth else None
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                if estimate_missing_from_color and ((want_depth and depth_np is None) or
                                                    (want_motion and motion_np is None)):
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(previous, rgba[..., :3], gpu_index=selected_gpu)
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth
                    if want_motion and motion_np is None:
                        motion_np = estimated_motion
                settings.reset = 1 if index == 0 else 0
                processed = bridge.process(rgba, settings, depth_np, motion_np)
                comparison = _comparison_rgba(rgba, processed)
                last_comparison = comparison
                processed_results.append(_tensor_rgb(processed))
                comparison_results.append(_tensor_rgb(comparison))
                previous = rgba[..., :3].copy()
                _check_interrupt()
            runtime_hash = bridge.runtime_hash
        elapsed = time.perf_counter() - started
        if last_comparison is not None:
            _preview(str(unique_id), last_comparison, batch,
                     batch / max(elapsed, 1e-6), runtime_hash, "stopped")
        status = (f"完成：{batch} 张，{elapsed:.2f}s，颜色估算设备={estimator_device}，"
                  f"DLL SHA256={runtime_hash}")
        return (torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device), status)


class DLSSNRProcessGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dll_path": runtime_input(),
                "gpu_device": gpu_input(),
                "automatic_mask": ("BOOLEAN", {"default": False}),
                "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
                "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "ui_correction": ("BOOLEAN", {"default": False}),
                "frame_guidance": (["0 使用可用引导（深度和运动）", "1 强制零引导", "2 仅运动向量", "3 仅深度"],),
                "depth_convention": (["1 反向深度（Inverted）", "0 正常深度（Normal）"],),
                "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "depth_inference_interval": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "estimate_missing_from_color": ("BOOLEAN", {"default": True}),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "preview_fps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "motion_vectors": ("DLSSNR_MOTION",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "comparison_image", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, image, dll_path, gpu_device, automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, ui_correction, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            depth_inference_interval, estimate_missing_from_color,
            motion_strength, preview_fps, unique_id=None, depth=None,
            motion_vectors=None):
        node_id = str(unique_id)
        batch, height, width, _ = image.shape
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y)
        processed_results = []
        comparison_results = []
        previous = None
        last_comparison = None
        started = time.perf_counter()
        last_preview = 0.0
        estimator_device = "unused"
        selected_gpu = gpu_index(gpu_device)
        with Bridge(dll_path, width, height, selected_gpu) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = (_depth_frame(depth, index, depth_inference_interval,
                                         height, width) if want_depth else None)
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                if estimate_missing_from_color and ((want_depth and depth_np is None) or
                                                    (want_motion and motion_np is None)):
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(previous, rgba[..., :3], motion_strength,
                                                1.0, 12, 960, selected_gpu)
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth
                    if want_motion and motion_np is None:
                        motion_np = estimated_motion
                settings.reset = 1 if index == 0 else 0
                processed = bridge.process(rgba, settings, depth_np, motion_np)
                last_comparison = _comparison_rgba(rgba, processed)
                processed_results.append(_tensor_rgb(processed))
                comparison_results.append(_tensor_rgb(last_comparison))
                previous = rgba[..., :3].copy()
                now = time.perf_counter()
                if now - last_preview >= 1.0 / preview_fps:
                    _preview(node_id, last_comparison, index + 1,
                             (index + 1) / max(now - started, 1e-6), bridge.runtime_hash)
                    last_preview = now
                _check_interrupt()
            elapsed = time.perf_counter() - started
            if last_comparison is not None:
                _preview(node_id, last_comparison, batch,
                         batch / max(elapsed, 1e-6), bridge.runtime_hash, "stopped")
            status = (f"完成：{batch} 帧，{elapsed:.2f}s，颜色估算设备={estimator_device}，"
                      f"DLL SHA256={bridge.runtime_hash}")
        return (torch.stack(processed_results).to(image.device),
                torch.stack(comparison_results).to(image.device), status)


class DLSSNRLoadGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gif_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "duration_ms")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"

    def run(self, gif_path):
        source = Path(gif_path.strip().strip('"')).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() != ".gif":
            raise RuntimeError(f"GIF 文件不存在：{source}")
        frames = []
        masks = []
        durations = []
        with Image.open(source) as gif:
            size = gif.size
            for frame in ImageSequence.Iterator(gif):
                rgba = frame.convert("RGBA")
                if rgba.size != size:
                    rgba = rgba.resize(size, Image.Resampling.LANCZOS)
                array = np.asarray(rgba, dtype=np.uint8)
                frames.append(torch.from_numpy(array[..., :3].copy()).float() / 255.0)
                masks.append(torch.from_numpy(array[..., 3].copy()).float() / 255.0)
                durations.append(int(frame.info.get("duration", 100) or 100))
        if not frames:
            raise RuntimeError("GIF 中没有可读取的帧。")
        return (torch.stack(frames), torch.stack(masks),
                max(10, int(round(float(np.mean(durations))))))


class DLSSNRSaveGIF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "DLSSNR_GIF", "multiline": False}),
                "duration_ms": ("INT", {"default": 100, "min": 10, "max": 10000, "step": 10}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 65535, "step": 1}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/GIF"
    OUTPUT_NODE = True

    def run(self, images, filename_prefix, duration_ms, loop_count, mask=None):
        frames = images.detach().float().cpu().clamp(0, 1).numpy()[..., :3]
        alpha = None if mask is None else mask.detach().float().cpu().clamp(0, 1).numpy()
        if alpha is not None and alpha.ndim == 4:
            alpha = alpha[..., 0]
        pil_frames = []
        for index, frame in enumerate(frames):
            rgb = np.ascontiguousarray(frame * 255.0 + 0.5, dtype=np.uint8)
            channel = (np.ascontiguousarray(alpha[min(index, len(alpha) - 1)] * 255.0 + 0.5,
                                             dtype=np.uint8) if alpha is not None else
                       np.full(rgb.shape[:2], 255, dtype=np.uint8))
            pil_frames.append(Image.fromarray(np.dstack((rgb, channel)), "RGBA"))
        prefix = Path(filename_prefix.strip().replace("\\", "/") or "DLSSNR_GIF")
        directory = _output_directory() / (prefix.parent if str(prefix.parent) != "." else "")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = _unique_path(directory / f"{prefix.name}_{stamp}.gif")
        pil_frames[0].save(output, save_all=True, append_images=pil_frames[1:],
                           duration=int(duration_ms), loop=int(loop_count), disposal=2)
        return (str(output), f"完成：保存 {len(pil_frames)} 帧 GIF。")


def _find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _has_nvenc(ffmpeg: str) -> bool:
    try:
        result = subprocess.run((ffmpeg, "-hide_banner", "-encoders"),
                                capture_output=True, text=True, timeout=15)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _mux_audio(ffmpeg: str | None, source: Path, video_only: Path,
               output: Path, preserve_audio: bool) -> bool:
    if preserve_audio and ffmpeg:
        for audio_args in (("-c:a", "copy"), ("-c:a", "aac", "-b:a", "192k")):
            command = (ffmpeg, "-y", "-loglevel", "error", "-i", str(video_only),
                       "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                       "-c:v", "copy", *audio_args, "-shortest", str(output))
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0 and output.is_file():
                video_only.unlink(missing_ok=True)
                return True
    os.replace(video_only, output)
    return False


class _StreamEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: float,
                 encoder_mode: str, gpu: int = 0):
        self.path = path
        self.process = None
        self.writer = None
        self.name = "CPU mp4v"
        self.ffmpeg = _find_ffmpeg()
        want_nvenc = str(encoder_mode).startswith("NVIDIA")
        if want_nvenc:
            if not self.ffmpeg or not _has_nvenc(self.ffmpeg):
                raise RuntimeError("未检测到可用的 ffmpeg h264_nvenc 编码器。请选择 CPU mp4v。")
            command = (self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                       "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                       "-r", f"{fps:.6f}", "-i", "pipe:0", "-an", "-c:v",
                       "h264_nvenc", "-gpu", str(max(0, int(gpu))),
                       "-preset", "p4", "-cq", "18", "-pix_fmt",
                       "yuv420p", str(path))
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE)
            self.name = "NVIDIA NVENC"
        else:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("CPU 视频编码需要 opencv-python。") from exc
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                          fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"无法创建视频输出：{path}")

    def write(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        if self.process:
            if self.process.stdin is None:
                raise RuntimeError("NVENC 输入管道不可用。")
            self.process.stdin.write(rgb.tobytes())
        else:
            import cv2
            self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.process:
            if self.process.stdin:
                self.process.stdin.close()
            error = self.process.stderr.read().decode("utf-8", "replace") if self.process.stderr else ""
            code = self.process.wait()
            if code != 0:
                raise RuntimeError(f"NVENC 编码失败：{error[-1200:]}")
        elif self.writer:
            self.writer.release()


class DLSSNRNativeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        gif_inputs = DLSSNRProcessGIF.INPUT_TYPES()
        required = dict(gif_inputs["required"])
        required.pop("image")
        required["frame_storage"] = (["GPU显存（VRAM）", "CPU内存（RAM）"],)
        required["max_frames"] = ("INT", {
            "default": 600, "min": 1, "max": 100000, "step": 1,
            "tooltip": "原生 VIDEO 接口会一次解码完整视频；超过此帧数时停止并提示改用流式节点。",
        })
        return {
            "required": {"video": ("VIDEO",), **required},
            "optional": dict(gif_inputs.get("optional", {})),
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "STRING")
    RETURN_NAMES = ("processed_video", "last_comparison", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, video, dll_path, gpu_device, automatic_mask, nr_style, nr_intensity,
            local_tone_strength, local_structure_strength,
            skin_structure_strength, ui_correction, frame_guidance,
            depth_convention, motion_scale_x, motion_scale_y,
            depth_inference_interval, estimate_missing_from_color,
            motion_strength, preview_fps, frame_storage, max_frames,
            unique_id=None, depth=None, motion_vectors=None):
        frame_count_fn = getattr(video, "get_frame_count", None)
        if callable(frame_count_fn):
            count = int(frame_count_fn())
            if count > int(max_frames):
                source_fn = getattr(video, "get_stream_source", None)
                source_value = source_fn() if callable(source_fn) else None
                if isinstance(source_value, (str, os.PathLike)):
                    ffmpeg = _find_ffmpeg()
                    encoder_mode = ("NVIDIA NVENC（GPU编码）"
                                    if ffmpeg and _has_nvenc(ffmpeg)
                                    else "CPU mp4v（CPU编码）")
                    fallback_guidance = (frame_guidance if estimate_missing_from_color
                                         else "1 强制零引导")
                    streamed = DLSSNRStreamVideo().run(
                        video_path="", output_path="", dll_path=dll_path,
                        gpu_device=gpu_device, encoder=encoder_mode,
                        processing_mode="最高速度处理", preserve_audio=True,
                        automatic_mask=automatic_mask, nr_style=nr_style,
                        nr_intensity=nr_intensity,
                        local_tone_strength=local_tone_strength,
                        local_structure_strength=local_structure_strength,
                        skin_structure_strength=skin_structure_strength,
                        ui_correction=ui_correction,
                        frame_guidance=fallback_guidance,
                        depth_convention=depth_convention,
                        motion_scale_x=motion_scale_x, motion_scale_y=motion_scale_y,
                        motion_strength=motion_strength, depth_strength=1.0,
                        flow_iterations=12, analysis_max_side=960,
                        preview_fps=preview_fps, unique_id=unique_id, video=video,
                        speed_multiplier=0.0,
                    )
                    return (streamed[0], streamed[2],
                            f"长视频已自动切换为逐帧低内存处理。{streamed[4]}")
                raise RuntimeError(
                    f"视频约有 {count} 帧，超过原生 VIDEO 节点上限 {max_frames}。"
                    "该 VIDEO 没有可流式读取的文件源，无法安全自动降级；"
                    "请使用文件型 VIDEO，或提高上限（可能耗尽内存/显存）。")
        components = video.get_components()
        image = components.images
        batch, height, width, _ = image.shape
        if batch > int(max_frames):
            raise RuntimeError(
                f"视频已解码 {batch} 帧，超过上限 {max_frames}。请改用低内存流式视频节点。")
        storage = (torch.device("cuda") if str(frame_storage).startswith("GPU")
                   and torch.cuda.is_available() else torch.device("cpu"))
        guidance = _guidance(frame_guidance)
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y)
        processed_results = []
        previous = None
        last_comparison = None
        started = time.perf_counter()
        last_preview = 0.0
        estimator_device = "unused"
        selected_gpu = gpu_index(gpu_device)
        with Bridge(dll_path, width, height, selected_gpu) as bridge:
            for index in range(batch):
                rgba = _rgba8(image[index])
                want_depth = guidance in (0, 3)
                want_motion = guidance in (0, 2)
                depth_np = (_depth_frame(depth, index, depth_inference_interval,
                                         height, width) if want_depth else None)
                motion_np = (_motion_frame(motion_vectors, index, height, width)
                             if want_motion else None)
                if estimate_missing_from_color and ((want_depth and depth_np is None) or
                                                    (want_motion and motion_np is None)):
                    estimated_depth, estimated_motion, _mask, _exposure, estimator_device = \
                        estimate_color_guidance(previous, rgba[..., :3], motion_strength,
                                                1.0, 12, 960, selected_gpu)
                    if want_depth and depth_np is None:
                        depth_np = estimated_depth
                    if want_motion and motion_np is None:
                        motion_np = estimated_motion
                settings.reset = 1 if index == 0 else 0
                processed = bridge.process(rgba, settings, depth_np, motion_np)
                last_comparison = _comparison_rgba(rgba, processed)
                processed_results.append(_tensor_rgb(processed, storage))
                previous = rgba[..., :3].copy()
                now = time.perf_counter()
                if now - last_preview >= 1.0 / preview_fps:
                    _preview(str(unique_id), last_comparison, index + 1,
                             (index + 1) / max(now - started, 1e-6), bridge.runtime_hash)
                    last_preview = now
                _check_interrupt()
            runtime_hash = bridge.runtime_hash
        images = torch.stack(processed_results)
        try:
            from comfy_api.latest import InputImpl, Types
        except ImportError:
            from comfy_api.v0_0_2 import InputImpl, Types
        video_components = Types.VideoComponents(
            images=images,
            audio=getattr(components, "audio", None),
            frame_rate=components.frame_rate,
        )
        output_video = InputImpl.VideoFromComponents(video_components)
        elapsed = time.perf_counter() - started
        _preview(str(unique_id), last_comparison, batch,
                 batch / max(elapsed, 1e-6), runtime_hash, "stopped")
        status = (f"完成：{batch} 帧，缓存={storage.type.upper()}，"
                  f"颜色估算={estimator_device.upper()}，DLL SHA256={runtime_hash}")
        return (output_video, _tensor_rgb(last_comparison, storage).unsqueeze(0), status)


class DLSSNRStreamVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": "", "multiline": False}),
            "output_path": ("STRING", {"default": "", "multiline": False}),
            "dll_path": runtime_input(),
            "gpu_device": gpu_input(),
            "encoder": (["NVIDIA NVENC（GPU编码）", "CPU mp4v（CPU编码）"],),
            "processing_mode": (["最高速度处理", "实时预览（按源帧率）"],),
            "preserve_audio": ("BOOLEAN", {"default": True}),
            "automatic_mask": ("BOOLEAN", {"default": False}),
            "nr_style": (["0 默认（Default）", "1 自然（Natural）", "2 电影感（Cinematic）"],),
            "nr_intensity": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "local_tone_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "local_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "skin_structure_strength": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
            "ui_correction": ("BOOLEAN", {"default": False}),
            "frame_guidance": (["0 从RGB估算深度和运动", "1 强制零引导", "2 仅从RGB估算运动", "3 仅从RGB估算深度"],),
            "depth_convention": (["1 反向深度（Inverted）", "0 正常深度（Normal）"],),
            "motion_scale_x": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "motion_scale_y": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
            "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "flow_iterations": ("INT", {"default": 12, "min": 1, "max": 50, "step": 1}),
            "analysis_max_side": ("INT", {"default": 960, "min": 128, "max": 4096, "step": 64}),
            "preview_fps": ("INT", {"default": 8, "min": 1, "max": 30, "step": 1}),
        }, "optional": {"video": ("VIDEO",)},
           "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("processed_video", "last_processed", "last_comparison",
                    "output_path", "status")
    FUNCTION = "run"
    CATEGORY = "image/DLSSNR Live/Video"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_):
        return float("nan")

    def run(self, video_path, output_path, dll_path, gpu_device, encoder,
            processing_mode, preserve_audio,
            automatic_mask, nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, skin_structure_strength, ui_correction,
            frame_guidance, depth_convention, motion_scale_x, motion_scale_y,
            motion_strength,
            depth_strength, flow_iterations, analysis_max_side, preview_fps,
            unique_id=None, video=None, speed_multiplier=0.0):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("流式视频解码需要 opencv-python。") from exc
        source_value = None
        if video is not None:
            source_fn = getattr(video, "get_stream_source", None)
            source_value = source_fn() if callable(source_fn) else None
            if source_value is not None and not isinstance(source_value, (str, os.PathLike)):
                raise RuntimeError(
                    "输入 VIDEO 没有磁盘文件源，无法低内存流式读取；"
                    "请改用「DLSSNR 原生视频处理（VIDEO→VIDEO）」节点。")
        source = Path(source_value if source_value is not None else
                      video_path.strip().strip('"')).expanduser().resolve()
        if not source.is_file():
            raise RuntimeError(f"视频文件不存在：{source}")
        requested = (Path(output_path.strip().strip('"')).expanduser().resolve()
                     if output_path.strip() else _output_directory() / f"{source.stem}_dlssnr.mp4")
        requested = _unique_path(requested.with_suffix(".mp4"))
        requested.parent.mkdir(parents=True, exist_ok=True)
        video_only = requested.with_name(f".{requested.stem}.video-only.mp4")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"无法打开视频：{source}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0.01 else 30.0
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        settings = _settings(
            nr_style, nr_intensity, local_tone_strength,
            local_structure_strength, automatic_mask, skin_structure_strength,
            ui_correction, int(str(depth_convention).split()[0]),
            motion_scale_x, motion_scale_y)
        guidance = _guidance(frame_guidance)
        event = _event_for(str(unique_id))
        writer = None
        previous = None
        last_processed = None
        last_comparison = None
        count = 0
        started = time.perf_counter()
        last_preview = 0.0
        estimator_device = "unused"
        decode_seconds = 0.0
        guidance_seconds = 0.0
        dlss_seconds = 0.0
        encode_seconds = 0.0
        selected_gpu = gpu_index(gpu_device)
        try:
            writer = _StreamEncoder(video_only, width, height, fps, encoder, selected_gpu)
            with Bridge(dll_path, width, height, selected_gpu) as bridge:
                while not event.is_set():
                    stage = time.perf_counter()
                    ok, bgr = capture.read()
                    decode_seconds += time.perf_counter() - stage
                    if not ok:
                        break
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgba = np.dstack((rgb, np.full((height, width), 255, dtype=np.uint8)))
                    depth_np = None
                    motion_np = None
                    if guidance != 1:
                        stage = time.perf_counter()
                        depth_est, motion_est, _mask, _exposure, estimator_device = \
                            estimate_color_guidance(previous, rgb, motion_strength,
                                                    depth_strength, flow_iterations,
                                                    analysis_max_side, selected_gpu)
                        guidance_seconds += time.perf_counter() - stage
                        if guidance in (0, 3):
                            depth_np = depth_est
                        if guidance in (0, 2):
                            motion_np = motion_est
                    settings.reset = 1 if count == 0 else 0
                    stage = time.perf_counter()
                    processed = bridge.process(rgba, settings, depth_np, motion_np)
                    dlss_seconds += time.perf_counter() - stage
                    last_processed = processed
                    last_comparison = _comparison_rgba(rgba, processed)
                    stage = time.perf_counter()
                    writer.write(processed[..., :3])
                    encode_seconds += time.perf_counter() - stage
                    previous = rgb.copy()
                    count += 1
                    now = time.perf_counter()
                    if now - last_preview >= 1.0 / preview_fps:
                        _preview(str(unique_id), last_comparison, count,
                                 count / max(now - started, 1e-6), bridge.runtime_hash)
                        last_preview = now
                    requested_speed = float(speed_multiplier)
                    pacing_speed = (1.0 if str(processing_mode).startswith("实时")
                                    else requested_speed if requested_speed > 0 else 0.0)
                    if pacing_speed > 0:
                        target = started + count / (fps * pacing_speed)
                        while not event.is_set():
                            delay = target - time.perf_counter()
                            if delay <= 0:
                                break
                            time.sleep(min(delay, 0.05))
                            _check_interrupt()
                    _check_interrupt()
                runtime_hash = bridge.runtime_hash
            writer.close()
            writer = None
            capture.release()
            ffmpeg = _find_ffmpeg()
            audio_kept = _mux_audio(ffmpeg, source, video_only, requested, preserve_audio)
            elapsed = time.perf_counter() - started
            if last_comparison is None or last_processed is None:
                raise RuntimeError("视频中没有可读取的帧。")
            _preview(str(unique_id), last_comparison, count,
                     count / max(elapsed, 1e-6), runtime_hash, "stopped")
            suffix = f"/{total}" if total else ""
            divisor = max(1, count)
            source_duration = count / max(fps, 1e-6)
            playback_speed = source_duration / max(elapsed, 1e-6)
            stages = (f"decode={decode_seconds * 1000 / divisor:.1f}ms，"
                      f"guide={guidance_seconds * 1000 / divisor:.1f}ms，"
                      f"DLSS={dlss_seconds * 1000 / divisor:.1f}ms，"
                      f"encode={encode_seconds * 1000 / divisor:.1f}ms")
            speed_limit_text = (f"{float(speed_multiplier):.2f}x"
                                if float(speed_multiplier) > 0 else "不限速")
            status = (f"完成：{count}{suffix} 帧，{elapsed:.2f}s，模式={processing_mode}，"
                      f"设定速度={speed_limit_text}，编码={encoder}，"
                      f"等效播放速度={playback_speed:.2f}x，"
                      f"颜色估算={estimator_device.upper()}，音频={'保留' if audio_kept else '未复用'}，"
                      f"{stages}，DLL SHA256={runtime_hash}")
            try:
                from comfy_api.latest import InputImpl
            except ImportError:
                from comfy_api.v0_0_2 import InputImpl
            output_video = InputImpl.VideoFromFile(str(requested))
            return (output_video, _tensor_rgb(last_processed).unsqueeze(0),
                    _tensor_rgb(last_comparison).unsqueeze(0), str(requested), status)
        finally:
            capture.release()
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            _remove_event(str(unique_id))


class DLSSNRFastVideo(DLSSNRStreamVideo):
    """Process/record frames as fast as possible, then preserve source timing."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = DLSSNRStreamVideo.INPUT_TYPES()
        required = dict(inputs["required"])
        required.pop("processing_mode", None)
        required["playback_speed"] = ("FLOAT", {
            "default": 2.0, "min": 0.0, "max": 64.0, "step": 0.25,
            "tooltip": "最高等效播放倍率；1=实时，2=两倍速，0=不限速。不会跳帧。",
        })
        return {
            "required": required,
            "optional": dict(inputs.get("optional", {})),
            "hidden": dict(inputs.get("hidden", {})),
        }

    def run(self, **kwargs):
        kwargs["processing_mode"] = "最高速度处理"
        kwargs["speed_multiplier"] = kwargs.pop("playback_speed", 2.0)
        return super().run(**kwargs)


NODE_CLASS_MAPPINGS = {
    "DLSSNRColorGuidance": DLSSNRColorGuidance,
    "DLSSNRImageDirect": DLSSNRImageDirect,
    "DLSSNRLoadGIF": DLSSNRLoadGIF,
    "DLSSNRProcessGIF": DLSSNRProcessGIF,
    "DLSSNRSaveGIF": DLSSNRSaveGIF,
    "DLSSNRNativeVideo": DLSSNRNativeVideo,
    "DLSSNRFastVideo": DLSSNRFastVideo,
    "DLSSNRStreamVideo": DLSSNRStreamVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSSNRColorGuidance": "DLSSNR RGB 引导估算（GPU）",
    "DLSSNRImageDirect": "DLSSNR 图像直接输出",
    "DLSSNRLoadGIF": "DLSSNR 加载 GIF",
    "DLSSNRProcessGIF": "DLSSNR GIF 处理",
    "DLSSNRSaveGIF": "DLSSNR 保存 GIF",
    "DLSSNRNativeVideo": "DLSSNR 原生视频处理（VIDEO→VIDEO）",
    "DLSSNRFastVideo": "DLSSNR GPU高速视频处理",
    "DLSSNRStreamVideo": "DLSSNR 流式视频处理（低内存）",
}
