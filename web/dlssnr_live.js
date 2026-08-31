import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function findNode(id) {
    return app.graph?.getNodeById?.(Number(id)) || app.graph?.getNodeById?.(id);
}

const PREVIEW_HEIGHT = 300;
const LIVE_PREVIEW_HEIGHT = 360;
const LIVE_PREVIEW_WIDTH = 1000;

const ZH_LABELS = {
    image: "图像", images: "图像帧", previous_image: "上一帧图像",
    processed_image: "处理后图像", comparison_image: "对比图像",
    last_processed: "最后处理帧", last_comparison: "最后对比帧",
    video: "视频", processed_video: "处理后视频", generated_video: "帧生成视频",
    output_path: "输出文件", media_path: "GIF/视频文件",
    video_path: "输入视频文件", gif_path: "GIF文件", duration_ms: "单帧时长（毫秒）",
    source_duration_ms: "原GIF单帧时长（毫秒）",
    duration_ms_out: "单帧时长（毫秒）", loop_count: "循环次数",
    filename_prefix: "文件名前缀", status: "状态", save_status: "保存状态",
    dll_path: "DLL文件", gpu_device: "GPU设备", encoder: "编码器",
    decoder: "解码器",
    processing_mode: "处理模式", playback_speed: "播放处理倍率",
    preserve_audio: "保留音频", run_mode: "运行模式",
    automatic_mask: "自动遮罩", nr_style: "降噪风格", nr_intensity: "降噪强度",
    nr_preset: "NR预设",
    local_tone_strength: "局部色调强度", local_structure_strength: "局部结构强度",
    skin_structure_strength: "皮肤结构强度", ui_correction: "UI修正",
    scene_paper_white_scale: "场景纸白亮度",
    frame_guidance: "帧引导方式", depth_convention: "深度方向",
    motion_scale_x: "运动向量X缩放", motion_scale_y: "运动向量Y缩放",
    depth_inference_interval: "深度推理间隔", estimate_missing_from_color: "从颜色估算缺失引导",
    motion_strength: "运动强度", depth_strength: "深度强度",
    flow_iterations: "光流迭代次数", analysis_max_side: "分析最大边长",
    depth_estimator: "深度估算方式", motion_estimator: "运动估算方式",
    enable_depth_estimation: "启用深度推理",
    depth_assist_strength: "节点侧深度辅助强度",
    normal_strength: "法线重建强度", normal_map: "估算法线图",
    edge_strength: "描边强度", edge_image: "描边图像", edge_mask: "边缘遮罩",
    motion_preview: "运动向量预览",
    enable_frame_generation: "启用GPU帧生成",
    frame_generation_multiplier: "帧生成倍率",
    generated_images: "帧生成图像",
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
    "depth_assist_strength",
]);

function scheduleHotSettings(node) {
    node._dlssnrHotPending = true;
    if (node._dlssnrHotInFlight || node._dlssnrHotFrame) return;
    node._dlssnrHotFrame = requestAnimationFrame(async () => {
        node._dlssnrHotFrame = null;
        if (!node._dlssnrHotPending) return;
        node._dlssnrHotPending = false;
        node._dlssnrHotInFlight = true;
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
        } finally {
            node._dlssnrHotInFlight = false;
            if (node._dlssnrHotPending) scheduleHotSettings(node);
        }
    });
}

function enableHotSettings(node) {
    for (const widget of node.widgets || []) {
        if (!HOT_WIDGET_NAMES.has(widget.name)) continue;
        if (widget._dlssnrHotSettingsHooked) continue;
        widget._dlssnrHotSettingsHooked = true;
        const originalCallback = widget.callback;
        widget.callback = function (value, ...rest) {
            const result = originalCallback?.call(this, value, ...rest);
            scheduleHotSettings(node);
            return result;
        };
    }
    if (!node._dlssnrWidgetChangedHooked) {
        node._dlssnrWidgetChangedHooked = true;
        const originalChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, ...rest) {
            const result = originalChanged?.call(this, name, value, ...rest);
            if (HOT_WIDGET_NAMES.has(name)) scheduleHotSettings(this);
            return result;
        };
    }
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

function openPreviewOverlay(node) {
    const source = node?._dlssnrLastPreview?.src;
    if (!source) {
        window.alert("当前还没有预览图，请先运行节点。");
        return;
    }
    const backdrop = document.createElement("div");
    Object.assign(backdrop.style, {
        position: "fixed", inset: "0", zIndex: "100000",
        background: "rgba(0,0,0,.92)", display: "flex",
        alignItems: "center", justifyContent: "center", cursor: "zoom-out",
    });
    const image = document.createElement("img");
    image.src = source;
    Object.assign(image.style, {
        maxWidth: "98vw", maxHeight: "94vh", objectFit: "contain",
        imageRendering: "auto", boxShadow: "0 0 0 1px rgba(255,255,255,.25)",
    });
    const hint = document.createElement("div");
    hint.textContent = "点击任意位置关闭 · 左侧原图 / 右侧处理结果";
    Object.assign(hint.style, {
        position: "fixed", left: "50%", bottom: "14px",
        transform: "translateX(-50%)", color: "white", fontSize: "14px",
        padding: "6px 12px", background: "rgba(0,0,0,.65)", borderRadius: "6px",
    });
    backdrop.append(image, hint);
    backdrop.addEventListener("click", () => backdrop.remove(), { once: true });
    document.body.appendChild(backdrop);
}

function uploadMedia(node, widgetName, kind) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = kind === "gif"
        ? "image/gif,.gif"
        : kind === "media"
            ? "image/gif,video/*,.gif,.mp4,.mkv,.mov,.avi,.webm,.m4v,.wmv"
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

function fitNodeToWidgets(node, nodeName) {
    const computed = node.computeSize?.() || [0, 0];
    const widgetFloor = 78 + (node.widgets?.length || 0) * 26;
    const isLive = nodeName === "DLSSNRLive";
    const minWidth = isLive ? LIVE_PREVIEW_WIDTH : 430;
    const previewHeight = isLive ? LIVE_PREVIEW_HEIGHT : PREVIEW_HEIGHT;
    const width = Math.max(Number(computed[0]) || 0, Number(node.size?.[0]) || 0, minWidth);
    const controlsHeight = Math.max(Number(computed[1]) || 0, widgetFloor);
    node._dlssnrControlsHeight = controlsHeight;
    const height = controlsHeight + previewHeight + 30;
    node.setSize?.([width, height]);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function ensureButton(node, label, callback) {
    const existing = (node.widgets || []).find((widget) =>
        widget.type === "button" && (widget.name === label || widget.label === label));
    if (existing) return existing;
    const button = node.addWidget("button", label, null, callback);
    button.serialize = false;
    return button;
}

function ensureNodeControls(node, nodeName, previewNodes) {
    if (["DLSSNRLive", "DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRFastVideoLegacy",
         "DLSSNRRealtimeStreamVideo"].includes(nodeName)) {
        ensureButton(node, "▶ 开始运行", async () => {
            await app.queuePrompt(0, 1);
        });
        ensureButton(node, "■ 停止并输出", async () => {
            await api.fetchApi("/dlssnr_live/stop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: String(node.id) }),
            });
        });
    }
    if (["DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRFastVideoLegacy",
         "DLSSNRRealtimeStreamVideo"].includes(nodeName)) {
        ensureButton(node, "选择输入视频", async () => {
            uploadMedia(node, "video_path", "video");
        });
    }
    if (nodeName === "DLSSNRLoadGIF") {
        ensureButton(node, "选择 GIF 文件", async () => {
            uploadMedia(node, "gif_path", "gif");
        });
    }
    if (nodeName === "DLSSNRFrameGenerateMedia") {
        ensureButton(node, "选择 GIF / 视频文件", async () => {
            uploadMedia(node, "media_path", "media");
        });
    }
    if (previewNodes.has(nodeName)) {
        ensureButton(node, "⛶ 全屏查看预览", () => openPreviewOverlay(node));
        fitNodeToWidgets(node, nodeName);
    }
}

function scheduleControlRepair(node, nodeName, previewNodes) {
    const repair = () => {
        translateNodeLabels(node);
        if (nodeName === "DLSSNRLive") enableHotSettings(node);
        ensureNodeControls(node, nodeName, previewNodes);
    };
    repair();
    requestAnimationFrame(repair);
    setTimeout(repair, 150);
    setTimeout(repair, 750);
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
            "DLSSNRLive", "DLSSNRImageDirect", "DLSSNRImageDirectLegacy", "DLSSNRProcessGIF",
            "DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRFastVideoLegacy", "DLSSNRNativeVideo", "DLSSNRLoadGIF",
            "DLSSNRFrameGenerateMedia", "DLSSNRRealtimeStreamVideo",
        ]);
        const previewNodes = new Set([
            "DLSSNRLive", "DLSSNRImageDirect", "DLSSNRImageDirectLegacy", "DLSSNRProcessGIF",
            "DLSSNRStreamVideo", "DLSSNRFastVideo", "DLSSNRFastVideoLegacy", "DLSSNRNativeVideo",
            "DLSSNRRealtimeStreamVideo",
        ]);
        if (!String(nodeData.name || "").startsWith("DLSSNR")) return;
        const hasExtraControls = supported.has(nodeData.name);
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            translateNodeLabels(this);
            requestAnimationFrame(() => translateNodeLabels(this));
            setTimeout(() => translateNodeLabels(this), 150);
            setTimeout(() => translateNodeLabels(this), 750);
            if (nodeData.name === "DLSSNRLive") enableHotSettings(this);
            if (hasExtraControls) {
                scheduleControlRepair(this, nodeData.name, previewNodes);
            }
        };
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            translateNodeLabels(this);
            requestAnimationFrame(() => translateNodeLabels(this));
            setTimeout(() => translateNodeLabels(this), 150);
            setTimeout(() => translateNodeLabels(this), 750);
            if (nodeData.name === "DLSSNRLive") enableHotSettings(this);
            if (hasExtraControls) {
                scheduleControlRepair(this, nodeData.name, previewNodes);
            }
            return result;
        };
        if (!previewNodes.has(nodeData.name)) {
            const originalTranslatedDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                translateNodeLabels(this);
                return originalTranslatedDraw?.apply(this, arguments);
            };
            return;
        }
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
            translateNodeLabels(this);
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
