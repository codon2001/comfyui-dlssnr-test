可以按照示例工作流来接线，dlss5的dll记得放到runtime里，下面都是ai写的。


## 节点

- `DLSSNR 实时预览 / 手动停止`：单张图片持续评估，节点下方实时显示处理前后的对比图。停止后第一个接口输出处理图，第二个接口输出对比图。
- `DLSSNR 图像直接输出`：图片或小批次执行完立即输出，不需要手动停止；同样可以输出处理图、对比图和状态。
- `DLSSNR 加载 GIF` / `DLSSNR GIF 处理` / `DLSSNR 保存 GIF`：保持可接线的 GIF 工作流，处理节点输出完整处理批次与完整对比批次。
- `DLSSNR视频处理相关节点`：还没改好，可以仿照示例工作流测试下。
- `DLSSNR RGB 引导估算（GPU）`：优先使用 PyTorch CUDA 从相邻 RGB 帧估算相对深度、当前帧到上一帧的运动向量、反应区域遮罩和曝光比。

## 推荐接线

图片实时：

```text
Load Image → DLSSNR 实时预览 / 手动停止
                   ├─ processed_image  → Save Image
                   └─ comparison_image → Save Image
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

GIF：

```text
DLSSNR 加载 GIF → DLSSNR GIF 处理 → DLSSNR 保存 GIF
                           ├─ processed_image
                           └─ comparison_image
```

## 参数

- `nr_style`：0 Default / 1 Natural / 2 Cinematic。
- `nr_intensity`、`local_tone_strength`、`local_structure_strength`、`skin_structure_strength`：范围 `-1.00～2.00`，默认 1.00。
- `automatic_mask`：运行库自动遮罩。
- `frame_guidance`：同时使用、强制零引导、仅运动、仅深度。
- `depth_convention`：反向深度或正常深度。
- `motion_scale_x/y`：运动向量缩放与方向，范围 `-4～4`。
- `scene_paper_white_scale`：场景纸白亮度，范围 `0.25～4.0`，默认 `1.0`；高于 1 提亮并使用保高光曲线，实时预览节点支持运行中滑块热调。它是 SDR/RGB 的等效颜色传递控制，不冒充 HDR 元数据或 NGX 原生参数。
- `depth_inference_interval`：批次中复用深度的帧间隔。
- `gpu_device`：选择负责 D3D11/D3D12处理与 CUDA颜色估算的 NVIDIA显卡；NVENC也使用同一序号。

## RGB估算

普通图片/视频无法获得游戏引擎内部的真实几何深度、物体运动向量、相机矩阵或投影 jitter。本包的颜色估算是后处理近似：

- 估算的 `depth` 和 `motion_vectors` 会真实传入 DLL。
- `reactive_mask` 与 `exposure_ratio` 当前作为诊断输出。
- 快速遮挡、透明物体、镜面、高曝光变化仍可能拖影或不稳定，不能等同于游戏原生引导数据。

## 安装

把整个文件夹放到 `ComfyUI/custom_nodes/`，安装 `requirements.txt` 后重启 ComfyUI。把其他兼容 DLL 放入包内 `runtimes` 或其子目录；重启后会自动出现在 `dll_path` 下拉列表。

