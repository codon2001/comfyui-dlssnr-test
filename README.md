# ComfyUI-DLSSNR-Live v16.2

Windows x64 / NVIDIA RTX 的独立 ComfyUI 节点包。图片、视频和 GIF 三条路径都会调用 `runtimes` 子目录中由用户选择的 DLSSNR DLL。

## 节点

- `DLSSNR 实时预览 / 手动停止`：单张图片持续评估，节点下方实时显示左右对比图。停止后第一个接口输出处理图，第二个接口输出对比图。
- `DLSSNR 图像直接输出`：图片或小批次执行完立即输出，不需要手动停止；同样输出处理图、对比图和状态。
- `DLSSNR 流式视频处理（低内存）`：支持文件型 `VIDEO` 输入和 `VIDEO` 输出，逐帧调用 GPU DLL并立即编码写盘，输出可直接连接 `Save Video`，同时保留源音频。
- `DLSSNR GPU高速视频处理`：主要视频节点，相当于把播放、连续 DLSSNR 处理和录制过程一起加速；`playback_speed` 可限制最高等效播放倍率（`1` 为实时、`2` 为两倍速、`0` 为不限速），所有档位均逐帧处理、不跳帧。最后按源 FPS 与原音频时间轴输出正常速度 VIDEO。状态中会显示设定倍率、实际等效倍率和各阶段耗时。
- `DLSSNR 原生视频处理（VIDEO→VIDEO）`：适合内存生成的视频或较短视频，可选择把处理帧缓存到 GPU显存或 CPU内存，并设置最大帧数保护。文件型 VIDEO 超过上限时会自动切换到逐帧低内存处理，避免一次展开全部帧；无文件源的内存视频仍保留安全上限。
- `DLSSNR 加载 GIF` / `DLSSNR GIF 处理` / `DLSSNR 保存 GIF`：保持可接线的 GIF 工作流，处理节点输出完整处理批次与完整对比批次。
- `DLSSNR RGB 引导估算（GPU）`：优先使用 PyTorch CUDA 从相邻 RGB 帧估算相对深度、当前帧到上一帧的运动向量、反应区域遮罩和曝光比。
- `DLSSNR 光流（Farneback）`：保留原有 `DLSSNR_MOTION` 输出，适合诊断或没有 CUDA 时使用。

## 推荐接线

图片实时：

```text
Load Image → DLSSNR 实时预览 / 手动停止
                   ├─ processed_image  → Save Image
                   └─ comparison_image → Save Image（可选）
```

图片直接输出：

```text
Load Image → DLSSNR 图像直接输出 → processed_image / comparison_image
```

外部或颜色估算引导：

```text
IMAGE → DLSSNR RGB 引导估算（GPU）
          ├─ estimated_depth → DLSSNR 节点的 depth
          └─ motion_vectors  → DLSSNR 节点的 motion_vectors
```

文件型视频推荐 `Load Video → DLSSNR GPU高速视频处理 → Save Video`。也可点击节点按钮使用浏览器文件选择器；媒体会复制到 `ComfyUI/input/dlssnr_uploads`。不要先把长视频展开成 `IMAGE` 批次。GPU编码需要本机 ffmpeg 提供 `h264_nvenc`；不可用时选择 `CPU mp4v`。内存生成的 VIDEO 使用原生视频节点。

文件型视频采用固定内存流式管线：每次只保留当前帧、上一帧、引导数据和最后预览，处理结果立即编码写盘；输出 VIDEO 是文件引用，内存/显存不会随视频总帧数持续累积。内存生成型 VIDEO 的整批张量由上游持有，受 `max_frames` 安全上限保护。

GIF：

```text
DLSSNR 加载 GIF → DLSSNR GIF 处理 → DLSSNR 保存 GIF
                           ├─ processed_image
                           └─ comparison_image
```

GIF 按 ComfyUI 批次工作，会占用与帧数成正比的内存，适合常规短 GIF；长动画建议转成视频后使用流式视频节点。

## 参数

- `nr_style`：0 Default / 1 Natural / 2 Cinematic。
- `nr_intensity`、`local_tone_strength`、`local_structure_strength`、`skin_structure_strength`：范围 `-1.00～2.00`，默认 1.00。界面与原生桥接使用相同范围。
- `automatic_mask`：运行库自动遮罩。
- `ui_correction`：UI correction。
- `frame_guidance`：同时使用、强制零引导、仅运动、仅深度。
- `depth_convention`：反向深度或正常深度。
- `motion_scale_x/y`：运动向量缩放与方向，范围 `-4～4`。
- `scene_paper_white_scale`：场景纸白亮度，范围 `0.25～4.0`，默认 `1.0`；高于 1 提亮并使用保高光曲线，实时预览节点支持运行中滑块热调。它是 SDR/RGB 的等效颜色传递控制，不冒充 HDR 元数据或 NGX 原生参数。
- `depth_inference_interval`：批次中复用深度的帧间隔。
- `gpu_device`：选择负责 D3D11/D3D12处理与 CUDA颜色估算的 NVIDIA显卡；NVENC也使用同一序号。

## RGB估算的边界

普通图片/视频无法获得游戏引擎内部的真实几何深度、物体运动向量、相机矩阵或投影 jitter。本包的颜色估算是后处理近似：

- 估算的 `depth` 和 `motion_vectors` 会真实传入 DLL。
- `reactive_mask` 与 `exposure_ratio` 当前作为诊断输出；桥接 ABI 没有对应外部资源入口，不会假装已经传入。
- 相机矩阵与 jitter 无法从普通 RGB 可靠恢复，因此不提供无效控件。
- 快速遮挡、透明物体、镜面、高曝光变化仍可能拖影或不稳定，不能等同于游戏原生引导数据。

## 安装

把整个文件夹放到 `ComfyUI/custom_nodes/ComfyUI-DLSSNR-Live`，安装 `requirements.txt` 后重启 ComfyUI，并在浏览器按 `Ctrl+F5`。把其他兼容 DLL 放入包内 `runtimes` 或其子目录；重启后会自动出现在 `dll_path` 下拉列表，不需要输入绝对路径。
