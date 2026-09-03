提醒，效果比游戏里用差很多

目前深度起到的效果依旧有限，等DLSS5的正式版出来看看再（

试着写了一些混合项，效果并不好，就先不发了

仓库不包含需要的模型文件，下载后请手动放置，或去releases下载完整包

可以按照示例工作流来接线，节点里记得重新选择下dll，工作流用的是我的电脑里的位置，图像和视频在我的5070Ti下处理得都很快

实时预览可以在运行过程中调节强度来测试合适的参数

下面都是ai写的


## 节点

- `DLSSNR 实时预览 / 手动停止`：持续处理单图，节点下方显示高清对比图；“全屏查看预览”可放大检查细节。
- `DLSSNR 图像直接输出`：一次运行后直接输出处理图、对比图和状态。
- `图像运算`：从 RGB 输出深度 IMAGE、运动向量、反应 MASK、曝光、法线 IMAGE、运动预览 IMAGE、描边 IMAGE 与描边 MASK。
- `DLSSNR GPU高速视频处理`：视频处理节点。
- GIF 加载/处理/保存：保持动画时长，可选择颜色深度、运动估算和 2–4 倍 GPU 光流补帧。

## 推荐接线

```text
IMAGE ─→ 图像运算
          ├─ estimated_depth ─→ 任一 DLSSNR 节点 depth / Preview Image
          ├─ motion_vectors  ─→ 任一 DLSSNR 节点 motion_vectors
          ├─ normal_map      ─→ Preview Image / Save Image / 其他 IMAGE 节点
          ├─ motion_preview  ─→ Preview Image / Save Image
          ├─ edge_image      ─→ Preview Image / Save Image
          └─ edge_mask       ─→ 其他 MASK 节点
```
其他连线请查看示例工作流


## 参数

- `nr_preset`：范围 `0～3`，默认 `0`。
- `nr_intensity`、局部色调/结构、皮肤结构：范围 `-1～2`。
- `depth_assist_strength`：默认 `1`。当前随包实验 DLL 对不同深度纹理的原生输出可能完全相同；此参数在节点侧按深度远近和边界调制 DLSSNR 局部增强。设为 `0` 可得到纯 DLL 原生路径。
- `scene_paper_white_scale`：范围 `0.25～4`，通过桥接层进行等效纸白亮度传递。
- `enable_depth_estimation`：关闭后不会运行深度模型。
- `轻量颜色深度`：快速的颜色/亮度相对深度，精度低于模型。
- `精细 GPU 迭代光流` 与 `高速 GPU 梯度光流`：真实参与运动输入和补帧；视频默认高速模式。
- `frame_guidance`：控制 DLL 接收深度、运动、两者或零引导。
- `enable_frame_generation`：在 GIF/视频节点内生成中间帧并提高输出帧率，同时保持播放时长。

普通媒体无法获得游戏引擎原生的几何深度、物体运动、曝光、反应遮罩、相机矩阵或投影 jitter。本包从 RGB 估算的引导会真实传入处理流程，但快速遮挡、透明/镜面对象和场景切换仍可能产生拖影。



## 安装

把整个文件夹放到 `ComfyUI/custom_nodes/`，若出现问题，请改文件夹名为COMFYUI-DLSSNR-TEST，安装 `requirements.txt` 后重启 ComfyUI。把其他兼容 DLL 放入包内 `runtimes` 或其子目录；重启后会自动出现在 `dll_path` 下拉列表。

