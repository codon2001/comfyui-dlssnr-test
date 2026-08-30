# ComfyUI-DLSSNR-Live

给 ComfyUI 用的 DLSSNR 画质增强节点，目前支持 **图片** 和 **GIF**（视频功能还在开发中）。

## 节点

**图片**

- `DLSSNR 实时预览 / 手动停止`：处理单张图片，节点下方实时显示处理前 / 处理后的左右对比。点停止后输出处理图和对比图。
- `DLSSNR 图像直接输出`：处理完立即出图，不用手动停止，同样输出处理图和对比图。

**GIF**

- `DLSSNR 加载 GIF` → `DLSSNR GIF 处理` → `DLSSNR 保存 GIF`：一条龙处理 GIF，输出处理结果和对比结果。适合普通短 GIF。

**可选引导**

- `DLSSNR RGB 引导估算（GPU）`：用 GPU 自动估算深度和运动信息，让增强效果更自然（需要 PyTorch CUDA）。

## 推荐接线

图片（实时预览）：

```text
Load Image → DLSSNR 实时预览 / 手动停止 → Save Image
```

图片（直接输出）：

```text
Load Image → DLSSNR 图像直接输出 → Save Image
```

GIF：

```text
DLSSNR 加载 GIF → DLSSNR GIF 处理 → DLSSNR 保存 GIF
```

## 主要参数

- `nr_style`：风格，`0` 默认 / `1` 自然 / `2` 电影感。
- `nr_intensity`、`local_tone_strength`、`local_structure_strength`、`skin_structure_strength`：强度类参数，范围 `-1.00～2.00`，默认 `1.00`，数值越大效果越强。
- `automatic_mask`：自动遮罩。
- `ui_correction`：界面矫正。
- `frame_guidance`：引导方式（同时使用 / 强制关闭 / 仅运动 / 仅深度）。
- `gpu_device`：选择用哪块 NVIDIA 显卡处理。

## 安装

把整个文件夹放到 `ComfyUI/custom_nodes/` 下，安装 `requirements.txt`，重启 ComfyUI，然后在浏览器里按 `Ctrl+F5` 刷新即可。
