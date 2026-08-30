import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function findNode(id) {
    return app.graph?.getNodeById?.(Number(id)) || app.graph?.getNodeById?.(id);
}

const PREVIEW_HEIGHT = 300;

const ZH_LABELS = {
    image: "图像", images: "图像帧", previous_image: "上一帧图像",
    processed_image: "处理后图像", comparison_image: "对比图像",
    last_processed: "最后处理帧", last_comparison: "最后对比帧",
    video: "视频", processed_video: "处理后视频", output_path: "输出文件",
    video_path: "输入视频文件", gif_path: "GIF文件", duration_ms: "单帧时长（毫秒）",
    duration_ms_out: "单帧时长（毫秒）", loop_count: "循环次数",
    filename_prefix: "文件名前缀", status: "状态", save_status: "保存状态",
    dll_path: "DLL文件", gpu_device: "GPU设备", encoder: "编码器",
    processing_mode: "处理模式", playback_speed: "播放处理倍率",
    preserve_audio: "保留音频", run_mode: "运行模式",
    automatic_mask: "自动遮罩", nr_style: "降噪风格", nr_intensity: "降噪强度",
    local_tone_strength: "局部色调强度", local_structure_strength: "局部结构强度",
    skin_structure_strength: "皮肤结构强度", ui_correction: "UI修正",
    scene_paper_white_scale: "场景纸白亮度",
    frame_guidance: "帧引导方式", depth_convention: "深度方向",
    motion_scale_x: "运动向量X缩放", motion_scale_y: "运动向量Y缩放",
    depth_inference_interval: "深度推理间隔", estimate_missing_from_color: "从颜色估算缺失引导",
    motion_strength: "运动强度", depth_strength: "深度强度",
    flow_iterations: "光流迭代次数", analysis_max_side: "分析最大边长",
    preview_fps: "预览帧率", safety_timeout_seconds: "安全超时（秒）",
    frame_storage: "帧缓存位置", max_frames: "最大帧数", strength: "强度",
    depth: "深度图", estimated_depth: "估算深度", motion_vectors: "运动向量",
    reactive_mask: "反应遮罩", exposure_ratio: "曝光比例", estimator_device: "估算设备",
    mask: "遮罩", output_file: "输出文件",
};

const HOT_WIDGET_NAMES = new Set([
    "automatic_mask", "nr_style", "nr_intensity", "local_tone_strength",
    "local_structure_strength", "skin_structure_strength", "ui_correction",
    "frame_guidance", "depth_convention", "motion_scale_x", "motion_scale_y",
    "depth_inference_interval", "preview_fps", "safety_timeout_seconds",
    "scene_paper_white_scale",
]);

function scheduleHotSettings(node) {
    clearTimeout(node._dlssnrHotTimer);
    node._dlssnrHotTimer = setTimeout(async () => {
        const settings = {};
        for (const widget of node.widgets || []) {
            if (HOT_WIDGET_NAMES.has(widget.name)) settings[widget.name] = widget.value;
        }
        try {
            const response = await api.fetchApi("/dlssnr_live/update_settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id), settings }),
            });
            const result = await response.json();
            if (result?.active) {
                node._dlssnrStatus = `运行中 · 参数已实时更新（版本 ${result.revision}）`;
                node.setDirtyCanvas?.(true, true);
            }
        } catch (error) {
            console.error("DLSSNR hot settings update failed", error);
        }
    }, 100);
}

function enableHotSettings(node) {
    if (node._dlssnrHotSettingsEnabled) return;
    node._dlssnrHotSettingsEnabled = true;
    for (const widget of node.widgets || []) {
        if (!HOT_WIDGET_NAMES.has(widget.name)) continue;
        const originalCallback = widget.callback;
        widget.callback = function (value, ...rest) {
            const result = originalCallback?.call(this, value, ...rest);
            scheduleHotSettings(node);
            return result;
        };
    }
    const originalChanged = node.onWidgetChanged;
    node.onWidgetChanged = function (name, value, ...rest) {
        const result = originalChanged?.call(this, name, value, ...rest);
        if (HOT_WIDGET_NAMES.has(name)) scheduleHotSettings(this);
        return result;
    };
}

function translateNodeLabels(node) {
    for (const widget of node.widgets || []) {
        const translated = ZH_LABELS[widget.name];
        if (translated) widget.label = translated;
    }
    for (const slot of node.inputs || []) {
        const translated = ZH_LABELS[slot.name];
        if (translated) slot.label = translated;
    }
    for (const slot of node.outputs || []) {
        const translated = ZH_LABELS[slot.name];
        if (translated) slot.label = translated;
    }
}

function showLocalPreview(source, image) {
    source._dlssnrLastPreview = image;
    source.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function uploadMedia(node, widgetName, kind) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = kind === "gif"
        ? "image/gif,.gif"
        : "video/*,.mp4,.mkv,.mov,.avi,.webm,.m4v,.wmv";
    input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) return;
        const data = new FormData();
        data.append("file", file, file.name);
        try {
            const response = await api.fetchApi(`/dlssnr_live/upload_media?kind=${kind}`, {
                method: "POST",
                body: data,
            });
            const result = await response.json();
            if (!response.ok || result?.error) throw new Error(result?.error || "上传失败");
            const widget = node.widgets?.find((item) => item.name === widgetName);
            if (widget) widget.value = result.path;
        } catch (error) {
            console.error("DLSSNR media upload failed", error);
            window.alert(`选择文件失败：${error?.message || error}`);
        }
    };
    input.click();
}

function fitNodeToWidgets(node) {
    const computed = node.computeSize?.() || [0, 0];
    const widgetFloor = 78 + (node.widgets?.length || 0) * 26;
    const width = Math.max(Number(computed[0]) || 0, Number(node.size?.[0]) || 0, 430);
    const controlsHeight = Math.max(Number(computed[1]) || 0, widgetFloor);
    node._dlssnrControlsHeight = controlsHeight;
    const height = controlsHeight + PREVIEW_HEIGHT + 30;
    node.setSize?.([width, height]);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

api.addEventListener("dlssnr_live_preview", (event) => {
    const data = event.detail;
    const node = findNode(data?.node_id);
    if (!node) return;
    const image = new Image();
    image.onload = () => {
        node._dlssnrStatus = `${data.state === "stopped" ? "已停止" : "运行中"} · ` +
            `${data.iteration} 次 · ${Number(data.fps || 0).toFixed(1)} FPS · ` +
            `SHA ${String(data.dll_sha256 || "").slice(0, 12)}`;
        showLocalPreview(node, image);
    };
    image.src = `data:image/jpeg;base64,${data.image}`;
});

app.registerExtension({
    name: "DLSSNR.Live.Preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const supported = new Set([
            "DLSSNRLive", "DLSSNRImageDirect", "DLSSNRProcessGIF",
            "DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRNativeVideo", "DLSSNRLoadGIF",
        ]);
        const previewNodes = new Set([
            "DLSSNRLive", "DLSSNRImageDirect", "DLSSNRProcessGIF",
            "DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRNativeVideo",
        ]);
        if (!String(nodeData.name || "").startsWith("DLSSNR")) return;
        const hasExtraControls = supported.has(nodeData.name);
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            translateNodeLabels(this);
            if (nodeData.name === "DLSSNRLive") enableHotSettings(this);
            if (!hasExtraControls) return;
            if (this._dlssnrControlsAdded) return;
            this._dlssnrControlsAdded = true;
            if (["DLSSNRLive", "DLSSNRStreamVideo", "DLSSNRFastVideo"].includes(nodeData.name)) {
                this.addWidget("button", "▶ 开始运行", null, async () => {
                    await app.queuePrompt(0, 1);
                });
                this.addWidget("button", "■ 停止并输出", null, async () => {
                    await api.fetchApi("/dlssnr_live/stop", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ node_id: String(this.id) }),
                    });
                });
            }
            if (["DLSSNRStreamVideo", "DLSSNRFastVideo"].includes(nodeData.name)) {
                this.addWidget("button", "选择输入视频", null, async () => {
                    uploadMedia(this, "video_path", "video");
                });
            }
            if (nodeData.name === "DLSSNRLoadGIF") {
                this.addWidget("button", "选择 GIF 文件", null, async () => {
                    uploadMedia(this, "gif_path", "gif");
                });
            }
            if (previewNodes.has(nodeData.name)) {
                fitNodeToWidgets(this);
                requestAnimationFrame(() => fitNodeToWidgets(this));
            }
        };
        if (!previewNodes.has(nodeData.name)) return;
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            requestAnimationFrame(() => fitNodeToWidgets(this));
            return result;
        };
        const originalBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            originalBackground?.apply(this, arguments);
            const image = this._dlssnrLastPreview;
            if (!image?.naturalWidth || !image?.naturalHeight) return;
            const x = 8;
            const y = (this._dlssnrControlsHeight || 0) + 8;
            const areaWidth = Math.max(1, this.size[0] - 16);
            const areaHeight = Math.max(0, this.size[1] - y - 30);
            if (areaWidth < 2 || areaHeight < 2) return;
            const scale = Math.min(areaWidth / image.naturalWidth, areaHeight / image.naturalHeight);
            const width = image.naturalWidth * scale;
            const height = image.naturalHeight * scale;
            const drawX = x + (areaWidth - width) / 2;
            const drawY = y + (areaHeight - height) / 2;
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, y, areaWidth, areaHeight);
            ctx.clip();
            ctx.fillStyle = "#111";
            ctx.fillRect(x, y, areaWidth, areaHeight);
            ctx.drawImage(image, drawX, drawY, width, height);
            ctx.restore();
        };
        const originalDraw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            originalDraw?.apply(this, arguments);
            if (!this._dlssnrStatus) return;
            ctx.save();
            ctx.beginPath();
            ctx.rect(6, this.size[1] - 26, Math.max(1, this.size[0] - 12), 20);
            ctx.clip();
            ctx.fillStyle = "rgba(0,0,0,.68)";
            ctx.fillRect(6, this.size[1] - 26, this.size[0] - 12, 20);
            ctx.fillStyle = "#fff";
            ctx.font = "12px sans-serif";
            ctx.fillText(this._dlssnrStatus, 12, this.size[1] - 12,
                         Math.max(1, this.size[0] - 24));
            ctx.restore();
        };
    },
});
