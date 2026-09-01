#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d10.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <codecapi.h>
#include <icodecapi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <process.h>
#include <string>
#include <thread>

#include "dlssnr_bridge.cpp"

namespace {

// MinGW headers omit this source-reader flag; it is 0x8 in the Windows SDK.
#ifndef MF_SOURCE_READERF_DATALOSS
#define MF_SOURCE_READERF_DATALOSS 0x8
#endif

constexpr GUID kCodecApiIid{
    0x901db4c7, 0x31ce, 0x41a2,
    {0x85, 0xdc, 0x8f, 0xa0, 0xbf, 0x41, 0xb8, 0xda}};

template<class T> void Release(T*& value) {
    if (value) { value->Release(); value = nullptr; }
}

std::string HrText(const char* stage, HRESULT hr) {
    char text[160]{};
    std::snprintf(text, sizeof(text), "%s failed (HRESULT 0x%08lX)",
                  stage, static_cast<unsigned long>(hr));
    return text;
}

int RunCpuFallback(int argc, wchar_t** argv, const std::string& reason) {
    wchar_t module[MAX_PATH]{};
    GetModuleFileNameW(nullptr, module, static_cast<DWORD>(std::size(module)));
    const auto fallback = std::filesystem::path(module).parent_path() /
        L"dlssnr_video_worker_cpu.exe";
    if (!std::filesystem::is_regular_file(fallback)) {
        std::fprintf(stderr, "ZERO_COPY_UNAVAILABLE %s; CPU fallback worker missing\n",
                     reason.c_str());
        return 20;
    }
    std::fprintf(stdout, "BACKEND_FALLBACK CPU_RGBA %s\n", reason.c_str());
    std::fflush(stdout);
    std::vector<const wchar_t*> arguments;
    arguments.reserve(static_cast<size_t>(argc) + 1);
    arguments.push_back(fallback.c_str());
    for (int i = 1; i < argc; ++i) arguments.push_back(argv[i]);
    arguments.push_back(nullptr);
    return static_cast<int>(_wspawnv(_P_WAIT, fallback.c_str(), arguments.data()));
}

void WriteGpuPreview(Session& session, ID3D11Texture2D* processedTexture,
                     const std::filesystem::path& path, UINT32 width, UINT32 height) {
    if (path.empty()) return;
    const UINT32 previewWidth = std::min<UINT32>(width, 480);
    const UINT32 previewHeight = std::max<UINT32>(1,
        static_cast<UINT32>(static_cast<uint64_t>(height) * previewWidth / width));
    std::vector<uint8_t> rgb(static_cast<size_t>(previewWidth * 2 + 4) * previewHeight * 3);
    auto copySide = [&](ID3D11Texture2D* texture, UINT32 side) {
        session.context11->CopyResource(session.staging11, texture);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (FAILED(session.context11->Map(session.staging11, 0, D3D11_MAP_READ, 0, &mapped)))
            return false;
        for (UINT32 y = 0; y < previewHeight; ++y) {
            const UINT32 sy = static_cast<UINT32>(static_cast<uint64_t>(y) * height / previewHeight);
            const auto* row = static_cast<const uint8_t*>(mapped.pData) +
                static_cast<size_t>(sy) * mapped.RowPitch;
            auto* destination = rgb.data() +
                (static_cast<size_t>(y) * (previewWidth * 2 + 4) + side * (previewWidth + 4)) * 3;
            for (UINT32 x = 0; x < previewWidth; ++x) {
                const UINT32 sx = static_cast<UINT32>(static_cast<uint64_t>(x) * width / previewWidth);
                const auto* pixel = row + sx * 4;
                destination[x * 3] = pixel[0];
                destination[x * 3 + 1] = pixel[1];
                destination[x * 3 + 2] = pixel[2];
            }
        }
        session.context11->Unmap(session.staging11, 0);
        return true;
    };
    if (!copySide(session.input11, 0) || !copySide(processedTexture, 1)) return;
    for (UINT32 y = 0; y < previewHeight; ++y) {
        auto* divider = rgb.data() +
            (static_cast<size_t>(y) * (previewWidth * 2 + 4) + previewWidth) * 3;
        const uint8_t values[12]{255,255,255,32,32,32,32,32,32,255,255,255};
        std::memcpy(divider, values, sizeof(values));
    }
    const auto temporary = std::filesystem::path(path.wstring() + L".tmp");
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) return;
    stream << "P6\n" << previewWidth * 2 + 4 << " " << previewHeight << "\n255\n";
    stream.write(reinterpret_cast<const char*>(rgb.data()),
                 static_cast<std::streamsize>(rgb.size()));
    stream.close();
    MoveFileExW(temporary.c_str(), path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

class GpuColorCorrector {
public:
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* sourceView = nullptr;
    ID3D11ShaderResourceView* outputView = nullptr;
    ID3D11UnorderedAccessView* correctedView = nullptr;
    ID3D11ComputeShader* shader = nullptr;
    ID3D11Buffer* constants = nullptr;
    UINT32 width = 0, height = 0;

    struct alignas(16) Parameters {
        float sourceMean, outputMean, lumaGain, chromaGain;
        float sourceChroma[4];
        float outputChroma[4];
        UINT32 width, height, padding[2];
    } parameters{};

    ~GpuColorCorrector() {
        Release(constants); Release(shader); Release(correctedView);
        Release(outputView); Release(sourceView); Release(texture);
    }

    bool Initialize(Session& session, std::string& error) {
        width = session.width; height = session.height;
        D3D11_TEXTURE2D_DESC desc{};
        session.output11->GetDesc(&desc);
        desc.MiscFlags = 0;
        HRESULT hr = session.device11Base->CreateTexture2D(&desc, nullptr, &texture);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateShaderResourceView(
            session.input11, nullptr, &sourceView);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateShaderResourceView(
            session.output11, nullptr, &outputView);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateUnorderedAccessView(
            texture, nullptr, &correctedView);
        static constexpr char source[] = R"(
Texture2D<float4> SourceImage : register(t0);
Texture2D<float4> EffectImage : register(t1);
RWTexture2D<float4> Corrected : register(u0);
cbuffer Params : register(b0) {
    float SourceMean; float OutputMean; float LumaGain; float ChromaGain;
    float4 SourceChroma; float4 OutputChroma;
    uint Width; uint Height; uint2 Padding;
};
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    float4 a = SourceImage[id.xy];
    // The D3D11 texture is already RGBA.  Swizzling it exchanges red/blue.
    float4 b = EffectImage[id.xy];
    const float3 c = float3(0.2126, 0.7152, 0.0722);
    float y = dot(b.rgb, c);
    float restoredY = (y - OutputMean) * LumaGain + SourceMean;
    float3 chroma = b.rgb - y;
    float3 rgb = restoredY + (chroma - OutputChroma.rgb) * ChromaGain + SourceChroma.rgb;
    Corrected[id.xy] = float4(saturate(rgb), a.a);
})";
        ID3DBlob* blob = nullptr;
        ID3DBlob* messages = nullptr;
        if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source) - 1, "GpuColorCorrection",
            nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
            &blob, &messages);
        if (FAILED(hr)) {
            error = messages ? static_cast<const char*>(messages->GetBufferPointer()) :
                HrText("Compile GPU color correction", hr);
        }
        Release(messages);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateComputeShader(
            blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader);
        Release(blob);
        D3D11_BUFFER_DESC bufferDesc{};
        bufferDesc.ByteWidth = sizeof(Parameters);
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateBuffer(
            &bufferDesc, nullptr, &constants);
        if (FAILED(hr)) { error = HrText("Create GPU color correction", hr); return false; }
        parameters.sourceMean = parameters.outputMean = 0.5f;
        parameters.lumaGain = parameters.chromaGain = 1.0f;
        parameters.width = width; parameters.height = height;
        // Upload a stable identity correction once.  Video must not jump to a
        // newly sampled global color transform every N frames.
        session.context11->UpdateSubresource(constants, 0, nullptr, &parameters, 0, 0);
        return true;
    }

    void UpdateStatistics(Session& session) {
        double mean[2]{}, square[2]{}, chroma[2][3]{}, chromaSquare[2]{};
        uint64_t count = 0;
        auto sample = [&](ID3D11Texture2D* image, int side) {
            session.context11->CopyResource(session.staging11, image);
            D3D11_MAPPED_SUBRESOURCE mapped{};
            if (FAILED(session.context11->Map(session.staging11, 0, D3D11_MAP_READ, 0, &mapped)))
                return false;
            constexpr double c[3]{0.2126, 0.7152, 0.0722};
            for (UINT32 y = 0; y < height; y += 4) {
                const auto* row = static_cast<const uint8_t*>(mapped.pData) +
                    static_cast<size_t>(y) * mapped.RowPitch;
                for (UINT32 x = 0; x < width; x += 4) {
                    const auto* pixel = row + x * 4;
                    const double values[3]{double(pixel[0]), double(pixel[1]),
                                           double(pixel[2])};
                    const double yv = (values[0] * c[0] + values[1] * c[1] + values[2] * c[2]) / 255.0;
                    mean[side] += yv; square[side] += yv * yv;
                    for (int channel = 0; channel < 3; ++channel) {
                        const double cv = values[channel] / 255.0 - yv;
                        chroma[side][channel] += cv;
                        chromaSquare[side] += cv * cv;
                    }
                    if (side == 0) ++count;
                }
            }
            session.context11->Unmap(session.staging11, 0);
            return true;
        };
        if (!sample(session.input11, 0) || !sample(session.output11, 1) || !count) return;
        for (int side = 0; side < 2; ++side) {
            mean[side] /= count; square[side] /= count;
            for (int channel = 0; channel < 3; ++channel) chroma[side][channel] /= count;
        }
        const double sourceStd = std::sqrt(std::max(square[0] - mean[0] * mean[0], 1e-8));
        const double outputStd = std::sqrt(std::max(square[1] - mean[1] * mean[1], 1e-8));
        parameters.sourceMean = static_cast<float>(mean[0]);
        parameters.outputMean = static_cast<float>(mean[1]);
        parameters.lumaGain = static_cast<float>(std::clamp(sourceStd / outputStd, 0.5, 2.0));
        double chromaVariance[2]{};
        for (int side = 0; side < 2; ++side) {
            chromaVariance[side] = chromaSquare[side] / count;
            for (int channel = 0; channel < 3; ++channel)
                chromaVariance[side] -= chroma[side][channel] * chroma[side][channel];
        }
        parameters.chromaGain = static_cast<float>(std::clamp(std::sqrt(
            std::max(chromaVariance[0], 1e-8) /
            std::max(chromaVariance[1], 1e-8)), 0.5, 2.0));
        for (int channel = 0; channel < 3; ++channel) {
            parameters.sourceChroma[channel] = static_cast<float>(chroma[0][channel]);
            parameters.outputChroma[channel] = static_cast<float>(chroma[1][channel]);
        }
        session.context11->UpdateSubresource(constants, 0, nullptr, &parameters, 0, 0);
    }

    void Apply(Session& session) {
        ID3D11ShaderResourceView* views[]{sourceView, outputView};
        session.context11->CSSetShader(shader, nullptr, 0);
        session.context11->CSSetShaderResources(0, 2, views);
        session.context11->CSSetUnorderedAccessViews(0, 1, &correctedView, nullptr);
        session.context11->CSSetConstantBuffers(0, 1, &constants);
        session.context11->Dispatch((width + 15) / 16, (height + 15) / 16, 1);
        ID3D11ShaderResourceView* nullViews[2]{};
        ID3D11UnorderedAccessView* nullUav = nullptr;
        session.context11->CSSetShaderResources(0, 2, nullViews);
        session.context11->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
        session.context11->CSSetShader(nullptr, nullptr, 0);
    }
};

class GpuMotionEstimator {
public:
    ID3D11Texture2D* previous = nullptr;
    ID3D11ShaderResourceView *currentView = nullptr, *previousView = nullptr;
    ID3D11UnorderedAccessView* motionView = nullptr;
    ID3D11ComputeShader* shader = nullptr;
    ID3D11Buffer* constants = nullptr;
    UINT32 width = 0, height = 0;
    bool hasPrevious = false;

    ~GpuMotionEstimator() {
        Release(constants); Release(shader); Release(motionView);
        Release(previousView); Release(currentView); Release(previous);
    }

    bool Initialize(Session& session, std::string& error) {
        width = session.width; height = session.height;
        D3D11_TEXTURE2D_DESC desc{};
        session.input11->GetDesc(&desc);
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        HRESULT hr = session.device11Base->CreateTexture2D(&desc, nullptr, &previous);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateShaderResourceView(
            session.input11, nullptr, &currentView);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateShaderResourceView(
            previous, nullptr, &previousView);
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateUnorderedAccessView(
            session.motion11, nullptr, &motionView);
        static constexpr char source[] = R"(
Texture2D<float4> Current : register(t0);
Texture2D<float4> Previous : register(t1);
RWTexture2D<float2> Motion : register(u0);
cbuffer Params : register(b0) { uint Width; uint Height; float Strength; float Limit; };
float Luma(int2 p, bool oldFrame) {
    p = clamp(p, int2(0,0), int2(Width-1, Height-1));
    float3 c = oldFrame ? Previous[p].rgb : Current[p].rgb;
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}
[numthreads(16,16,1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    int2 p = int2(id.xy);
    float ix = (Luma(p+int2(1,0),false)-Luma(p-int2(1,0),false)
              + Luma(p+int2(1,0),true)-Luma(p-int2(1,0),true)) * 0.25;
    float iy = (Luma(p+int2(0,1),false)-Luma(p-int2(0,1),false)
              + Luma(p+int2(0,1),true)-Luma(p-int2(0,1),true)) * 0.25;
    float it = Luma(p,false)-Luma(p,true);
    float denom = ix*ix + iy*iy + 0.0005;
    Motion[p] = clamp(-it*float2(ix,iy)/denom*Strength, -Limit, Limit);
})";
        ID3DBlob *blob = nullptr, *messages = nullptr;
        if (SUCCEEDED(hr)) hr = D3DCompile(source, sizeof(source)-1, "GpuMotion",
            nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0, &blob, &messages);
        if (FAILED(hr)) {
            error = messages ? static_cast<const char*>(messages->GetBufferPointer()) :
                HrText("Compile GPU motion estimator", hr);
            Release(messages); return false;
        }
        Release(messages);
        hr = session.device11Base->CreateComputeShader(blob->GetBufferPointer(),
            blob->GetBufferSize(), nullptr, &shader);
        Release(blob);
        D3D11_BUFFER_DESC bd{}; bd.ByteWidth = 16; bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (SUCCEEDED(hr)) hr = session.device11Base->CreateBuffer(&bd, nullptr, &constants);
        if (FAILED(hr)) { error = HrText("Create GPU motion estimator", hr); return false; }
        return true;
    }

    void Run(Session& session, float strength) {
        const float zero[4]{};
        if (!hasPrevious) {
            session.context11->ClearUnorderedAccessViewFloat(motionView, zero);
        } else {
            struct Params { UINT32 width, height; float strength, limit; };
            Params p{width, height, strength, 32.0f};
            session.context11->UpdateSubresource(constants, 0, nullptr, &p, 0, 0);
            ID3D11ShaderResourceView* views[]{currentView, previousView};
            session.context11->CSSetShader(shader, nullptr, 0);
            session.context11->CSSetShaderResources(0, 2, views);
            session.context11->CSSetUnorderedAccessViews(0, 1, &motionView, nullptr);
            session.context11->CSSetConstantBuffers(0, 1, &constants);
            session.context11->Dispatch((width+15)/16, (height+15)/16, 1);
            ID3D11ShaderResourceView* nullViews[2]{};
            ID3D11UnorderedAccessView* nullUav = nullptr;
            session.context11->CSSetShaderResources(0, 2, nullViews);
            session.context11->CSSetUnorderedAccessViews(0, 1, &nullUav, nullptr);
            session.context11->CSSetShader(nullptr, nullptr, 0);
        }
        session.context11->CopyResource(previous, session.input11);
        hasPrevious = true;
        session.zeroGuidanceInitialized = true;
    }
};

class GpuDepthEstimator {
public:
    ID3D11ShaderResourceView* sourceView = nullptr;
    ID3D11UnorderedAccessView* depthView = nullptr;
    ID3D11ComputeShader* shader = nullptr;
    ID3D11Buffer* constants = nullptr;
    UINT32 width = 0, height = 0;
    ~GpuDepthEstimator() { Release(constants); Release(shader); Release(depthView); Release(sourceView); }
    bool Initialize(Session& session, std::string& error) {
        width=session.width; height=session.height;
        HRESULT hr=session.device11Base->CreateShaderResourceView(session.input11,nullptr,&sourceView);
        if(SUCCEEDED(hr)) hr=session.device11Base->CreateUnorderedAccessView(session.depth11,nullptr,&depthView);
        static constexpr char source[]=R"(
Texture2D<float4> Input:register(t0);RWTexture2D<float> Depth:register(u0);
cbuffer P:register(b0){uint Width;uint Height;float Strength;float Pad;};
[numthreads(16,16,1)]void main(uint3 id:SV_DispatchThreadID){
 if(id.x>=Width||id.y>=Height)return;int2 p=int2(id.xy);
 float y=dot(Input[p].rgb,float3(.2126,.7152,.0722));
 int2 px=int2(min(id.x+1,Width-1),id.y),py=int2(id.x,min(id.y+1,Height-1));
 float ex=abs(y-dot(Input[px].rgb,float3(.2126,.7152,.0722)));
 float ey=abs(y-dot(Input[py].rgb,float3(.2126,.7152,.0722)));
 Depth[p]=saturate(((1-y)*.82+(ex+ey)*2.0)*Strength);
})";
        ID3DBlob *blob=nullptr,*messages=nullptr;
        if(SUCCEEDED(hr)) hr=D3DCompile(source,sizeof(source)-1,"GpuDepth",nullptr,nullptr,
            "main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&messages);
        if(FAILED(hr)){error=messages?static_cast<const char*>(messages->GetBufferPointer()):
            HrText("Compile GPU depth estimator",hr);Release(messages);return false;}
        Release(messages);hr=session.device11Base->CreateComputeShader(blob->GetBufferPointer(),
            blob->GetBufferSize(),nullptr,&shader);Release(blob);
        D3D11_BUFFER_DESC bd{};bd.ByteWidth=16;bd.Usage=D3D11_USAGE_DEFAULT;
        bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(SUCCEEDED(hr))hr=session.device11Base->CreateBuffer(&bd,nullptr,&constants);
        if(FAILED(hr)){error=HrText("Create GPU depth estimator",hr);return false;}return true;
    }
    void Run(Session& session,float strength){
        struct P{UINT32 width,height;float strength,pad;}p{width,height,strength,0};
        session.context11->UpdateSubresource(constants,0,nullptr,&p,0,0);
        session.context11->CSSetShader(shader,nullptr,0);
        session.context11->CSSetShaderResources(0,1,&sourceView);
        session.context11->CSSetUnorderedAccessViews(0,1,&depthView,nullptr);
        session.context11->CSSetConstantBuffers(0,1,&constants);
        session.context11->Dispatch((width+15)/16,(height+15)/16,1);
        ID3D11ShaderResourceView* nullView=nullptr;ID3D11UnorderedAccessView* nullUav=nullptr;
        session.context11->CSSetShaderResources(0,1,&nullView);
        session.context11->CSSetUnorderedAccessViews(0,1,&nullUav,nullptr);
        session.context11->CSSetShader(nullptr,nullptr,0);
        session.zeroGuidanceInitialized=true;
    }
};

bool SetSize(IMFMediaType* type, REFGUID key, UINT32 width, UINT32 height) {
    return SUCCEEDED(MFSetAttributeSize(type, key, width, height));
}

bool SetRatio(IMFMediaType* type, REFGUID key, UINT32 numerator, UINT32 denominator) {
    return SUCCEEDED(MFSetAttributeRatio(type, key, numerator, denominator));
}

class GpuVideoPipeline {
public:
    IMFSourceReader* reader = nullptr;
    IMFSinkWriter* writer = nullptr;
    IMFDXGIDeviceManager* manager = nullptr;
    ID3D11Device* d3dDevice = nullptr;
    ID3D11DeviceContext* d3dContext = nullptr;
    ID3D11VideoDevice* videoDevice = nullptr;
    ID3D11VideoContext* videoContext = nullptr;
    ID3D11VideoProcessorEnumerator* enumerator = nullptr;
    ID3D11VideoProcessor* processor = nullptr;
    ID3D11VideoProcessorOutputView* dlssInputView = nullptr;
    ID3D11Texture2D* encodeTexture = nullptr;
    ID3D11VideoProcessorInputView* dlssOutputView = nullptr;
    ID3D11VideoProcessorOutputView* encodeView = nullptr;
    DWORD sinkStream = 0;
    UINT resetToken = 0;
    UINT32 width = 0, height = 0;
    UINT32 fpsNum = 60, fpsDen = 1;
    UINT decodeFrameIndex = 0;
    UINT encodeFrameIndex = 0;
    uint64_t timelineFrameIndex = 0;
    LONGLONG frameDuration = 166667;
    uint64_t dataLossSamples = 0;
    uint64_t outOfOrderSamples = 0;

    LONGLONG FrameTime(uint64_t index) const {
        const long double ticks = static_cast<long double>(index) * 10000000.0L *
            static_cast<long double>(fpsDen) / static_cast<long double>(fpsNum);
        return static_cast<LONGLONG>(std::llround(ticks));
    }

    void ConfigureVideoProcessor(bool encode) {
        const RECT full{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        D3D11_VIDEO_PROCESSOR_COLOR_SPACE limited709{};
        limited709.Usage = 0;
        limited709.RGB_Range = 1;
        limited709.YCbCr_Matrix = 1;
        limited709.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235;
        D3D11_VIDEO_PROCESSOR_COLOR_SPACE fullRgb{};
        fullRgb.Usage = 0;
        fullRgb.RGB_Range = 0;
        fullRgb.YCbCr_Matrix = 1;
        fullRgb.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255;
        videoContext->VideoProcessorSetStreamFrameFormat(
            processor, 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        videoContext->VideoProcessorSetStreamAutoProcessingMode(processor, 0, FALSE);
        videoContext->VideoProcessorSetStreamSourceRect(processor, 0, TRUE, &full);
        videoContext->VideoProcessorSetStreamDestRect(processor, 0, TRUE, &full);
        videoContext->VideoProcessorSetOutputTargetRect(processor, TRUE, &full);
        videoContext->VideoProcessorSetStreamColorSpace(
            processor, 0, encode ? &fullRgb : &limited709);
        videoContext->VideoProcessorSetOutputColorSpace(
            processor, encode ? &limited709 : &fullRgb);
    }

    ~GpuVideoPipeline() { Destroy(); }

    bool Finish(std::string& error) {
        if (!writer) return true;
        const HRESULT hr = writer->Finalize();
        Release(writer);
        if (FAILED(hr)) { error = HrText("Finalize GPU video", hr); return false; }
        return true;
    }

    void Destroy() {
        if (writer) writer->Finalize();
        Release(encodeView); Release(dlssOutputView); Release(encodeTexture);
        Release(dlssInputView); Release(processor); Release(enumerator);
        Release(videoContext); Release(videoDevice); Release(d3dContext);
        Release(writer); Release(reader); Release(manager); Release(d3dDevice);
    }

    bool Initialize(Session& session, const wchar_t* input, const wchar_t* output,
                    UINT32 w, UINT32 h, UINT32 rateNum, UINT32 rateDen,
                    std::string& error) {
        width = w; height = h; fpsNum = std::max(1u, rateNum); fpsDen = std::max(1u, rateDen);
        frameDuration = static_cast<LONGLONG>(10000000.0 * fpsDen / fpsNum + 0.5);
        HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_FULL);
        if (FAILED(hr)) { error = HrText("MFStartup", hr); return false; }
        hr = MFCreateDXGIDeviceManager(&resetToken, &manager);
        if (SUCCEEDED(hr)) hr = manager->ResetDevice(session.device11Base, resetToken);
        if (FAILED(hr)) { error = HrText("Create DXGI device manager", hr); return false; }
        d3dDevice = session.device11Base;
        d3dDevice->AddRef();
        d3dContext = session.context11Base;
        d3dContext->AddRef();
        ID3D10Multithread* multithread = nullptr;
        if (SUCCEEDED(session.device11Base->QueryInterface(IID_ID3D10Multithread,
                reinterpret_cast<void**>(&multithread)))) {
            multithread->SetMultithreadProtected(TRUE);
            multithread->Release();
        }

        IMFAttributes* attributes = nullptr;
        hr = MFCreateAttributes(&attributes, 6);
        if (SUCCEEDED(hr)) hr = attributes->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, manager);
        if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
        if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, FALSE);
        if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_SOURCE_READER_DISCONNECT_MEDIASOURCE_ON_SHUTDOWN, TRUE);
        if (SUCCEEDED(hr)) hr = MFCreateSourceReaderFromURL(input, attributes, &reader);
        Release(attributes);
        if (FAILED(hr)) { error = HrText("Create hardware source reader", hr); return false; }
        IMFMediaType* decodeType = nullptr;
        hr = MFCreateMediaType(&decodeType);
        if (SUCCEEDED(hr)) hr = decodeType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) hr = decodeType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        if (SUCCEEDED(hr)) hr = reader->SetCurrentMediaType(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, decodeType);
        Release(decodeType);
        if (FAILED(hr)) { error = HrText("Select NV12 hardware decoder output", hr); return false; }
        reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
        reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);

        hr = session.device11Base->QueryInterface(IID_ID3D11VideoDevice,
            reinterpret_cast<void**>(&videoDevice));
        if (SUCCEEDED(hr)) hr = session.context11Base->QueryInterface(IID_ID3D11VideoContext,
            reinterpret_cast<void**>(&videoContext));
        if (FAILED(hr)) { error = HrText("Query D3D11 video interfaces", hr); return false; }
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content{};
        content.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content.InputFrameRate = {fpsNum, fpsDen};
        content.InputWidth = width; content.InputHeight = height;
        content.OutputFrameRate = {fpsNum, fpsDen};
        content.OutputWidth = width; content.OutputHeight = height;
        content.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        hr = videoDevice->CreateVideoProcessorEnumerator(&content, &enumerator);
        UINT rgbaFlags = 0, nv12Flags = 0;
        if (SUCCEEDED(hr)) hr = enumerator->CheckVideoProcessorFormat(
            DXGI_FORMAT_R8G8B8A8_UNORM, &rgbaFlags);
        if (SUCCEEDED(hr)) hr = enumerator->CheckVideoProcessorFormat(
            DXGI_FORMAT_NV12, &nv12Flags);
        if (FAILED(hr) || !(rgbaFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) ||
            !(rgbaFlags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT) ||
            !(nv12Flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) ||
            !(nv12Flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT)) {
            error = "GPU video processor does not support NV12 <-> RGBA8";
            return false;
        }
        hr = videoDevice->CreateVideoProcessor(enumerator, 0, &processor);
        if (FAILED(hr)) { error = HrText("Create video processor", hr); return false; }
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC outputViewDesc{};
        outputViewDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        hr = videoDevice->CreateVideoProcessorOutputView(
            session.input11, enumerator, &outputViewDesc, &dlssInputView);
        if (FAILED(hr)) { error = HrText("Create DLSS input output-view", hr); return false; }
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC outputInputDesc{};
        outputInputDesc.FourCC = 0;
        outputInputDesc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        outputInputDesc.Texture2D.MipSlice = 0;
        outputInputDesc.Texture2D.ArraySlice = 0;
        hr = videoDevice->CreateVideoProcessorInputView(
            session.output11, enumerator, &outputInputDesc, &dlssOutputView);
        if (FAILED(hr)) { error = HrText("Create DLSS output input-view", hr); return false; }

        hr = MFCreateAttributes(&attributes, 5);
        if (SUCCEEDED(hr)) hr = attributes->SetUnknown(MF_SINK_WRITER_D3D_MANAGER, manager);
        if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
        if (SUCCEEDED(hr)) hr = attributes->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, TRUE);
        if (SUCCEEDED(hr)) hr = MFCreateSinkWriterFromURL(output, nullptr, attributes, &writer);
        Release(attributes);
        if (FAILED(hr)) { error = HrText("Create hardware sink writer", hr); return false; }
        IMFMediaType* encoded = nullptr;
        hr = MFCreateMediaType(&encoded);
        if (SUCCEEDED(hr)) hr = encoded->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) hr = encoded->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        const uint64_t qualityBitrate = std::clamp<uint64_t>(
            static_cast<uint64_t>(width) * height * fpsNum / fpsDen / 4,
            20000000ull, 200000000ull);
        if (SUCCEEDED(hr)) hr = encoded->SetUINT32(
            MF_MT_AVG_BITRATE, static_cast<UINT32>(qualityBitrate));
        if (SUCCEEDED(hr)) hr = encoded->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        if (SUCCEEDED(hr) && !SetSize(encoded, MF_MT_FRAME_SIZE, width, height)) hr = E_FAIL;
        if (SUCCEEDED(hr) && !SetRatio(encoded, MF_MT_FRAME_RATE, fpsNum, fpsDen)) hr = E_FAIL;
        if (SUCCEEDED(hr) && !SetRatio(encoded, MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) hr = E_FAIL;
        if (SUCCEEDED(hr)) hr = writer->AddStream(encoded, &sinkStream);
        Release(encoded);
        IMFMediaType* raw = nullptr;
        if (SUCCEEDED(hr)) hr = MFCreateMediaType(&raw);
        if (SUCCEEDED(hr)) hr = raw->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) hr = raw->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        if (SUCCEEDED(hr)) hr = raw->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        if (SUCCEEDED(hr) && !SetSize(raw, MF_MT_FRAME_SIZE, width, height)) hr = E_FAIL;
        if (SUCCEEDED(hr) && !SetRatio(raw, MF_MT_FRAME_RATE, fpsNum, fpsDen)) hr = E_FAIL;
        if (SUCCEEDED(hr) && !SetRatio(raw, MF_MT_PIXEL_ASPECT_RATIO, 1, 1)) hr = E_FAIL;
        if (SUCCEEDED(hr)) hr = writer->SetInputMediaType(sinkStream, raw, nullptr);
        Release(raw);
        ICodecAPI* codec = nullptr;
        if (SUCCEEDED(hr) && SUCCEEDED(writer->GetServiceForStream(
                sinkStream, GUID_NULL, kCodecApiIid,
                reinterpret_cast<void**>(&codec)))) {
            auto setU32 = [&](const GUID& key, ULONG value) {
                VARIANT option{};
                option.vt = VT_UI4;
                option.ulVal = value;
                codec->SetValue(&key, &option);
            };
            auto setBool = [&](const GUID& key, bool value) {
                VARIANT option{};
                option.vt = VT_BOOL;
                option.boolVal = value ? VARIANT_TRUE : VARIANT_FALSE;
                codec->SetValue(&key, &option);
            };
            setU32(CODECAPI_AVEncCommonRateControlMode,
                   eAVEncCommonRateControlMode_Quality);
            setU32(CODECAPI_AVEncCommonQuality, 90);
            setU32(CODECAPI_AVEncCommonMeanBitRate,
                   static_cast<ULONG>(qualityBitrate));
            setBool(CODECAPI_AVEncCommonAllowFrameDrops, false);
            setBool(CODECAPI_AVEncCommonLowLatency, false);
            codec->Release();
        }
        if (SUCCEEDED(hr)) hr = writer->BeginWriting();
        if (FAILED(hr)) { error = HrText("Configure hardware H.264 writer", hr); return false; }
        return true;
    }

    bool SetEncodeInput(ID3D11Texture2D* texture, std::string& error) {
        Release(dlssOutputView);
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC desc{};
        desc.FourCC = 0;
        desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipSlice = 0;
        desc.Texture2D.ArraySlice = 0;
        const HRESULT hr = videoDevice->CreateVideoProcessorInputView(
            texture, enumerator, &desc, &dlssOutputView);
        if (FAILED(hr)) { error = HrText("Select corrected GPU encode input", hr); return false; }
        return true;
    }

    bool ReadTexture(ID3D11Texture2D** texture, UINT* subresource,
                     LONGLONG* timestamp, IMFSample** outSample, bool& eof,
                     std::string& error) {
        *texture = nullptr; *subresource = 0; eof = false; *outSample = nullptr;
        for (;;) {
            DWORD actual = 0, flags = 0;
            IMFSample* sample = nullptr;
            HRESULT hr = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
                &actual, &flags, timestamp, &sample);
            if (FAILED(hr)) { error = HrText("Read hardware decoded frame", hr); return false; }
            if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { Release(sample); eof = true; return true; }
            if (flags & MF_SOURCE_READERF_DATALOSS) ++dataLossSamples;
            if (!sample) continue;
            IMFMediaBuffer* buffer = nullptr;
            hr = sample->GetBufferByIndex(0, &buffer);
            IMFDXGIBuffer* dxgi = nullptr;
            if (SUCCEEDED(hr)) hr = buffer->QueryInterface(IID_IMFDXGIBuffer,
                reinterpret_cast<void**>(&dxgi));
            if (SUCCEEDED(hr)) hr = dxgi->GetResource(IID_ID3D11Texture2D,
                reinterpret_cast<void**>(texture));
            if (SUCCEEDED(hr)) hr = dxgi->GetSubresourceIndex(subresource);
            Release(dxgi); Release(buffer);
            if (FAILED(hr)) {
                Release(sample);
                error = "Decoder returned a CPU frame; zero-copy NV12 surface unavailable";
                return false;
            }
            // The decoded DXGI surface belongs to the decoder's surface pool.
            // Releasing the sample here returns the surface to that pool while
            // the asynchronous VideoProcessorBlt below is still queued, so the
            // decoder can overwrite it with a newer frame before the blit
            // executes - the pipeline then briefly processes a future frame,
            // which appears as a visible flash/stutter in the encoded video.
            // Keep the sample alive until the blit has been consumed (the
            // caller releases it after ProcessGpuTexture returned).
            *outSample = sample;
            return true;
        }
    }

    bool Nv12ToDlssInput(ID3D11Texture2D* texture, UINT subresource, std::string& error) {
        D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC desc{};
        desc.FourCC = 0; desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
        desc.Texture2D.MipSlice = 0; desc.Texture2D.ArraySlice = subresource;
        ID3D11VideoProcessorInputView* view = nullptr;
        HRESULT hr = videoDevice->CreateVideoProcessorInputView(texture, enumerator, &desc, &view);
        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        ConfigureVideoProcessor(false);
        stream.Enable = TRUE;
        stream.InputFrameOrField = decodeFrameIndex;
        stream.pInputSurface = view;
        if (SUCCEEDED(hr)) hr = videoContext->VideoProcessorBlt(
            processor, dlssInputView, decodeFrameIndex, 1, &stream);
        Release(view);
        if (FAILED(hr)) { error = HrText("NV12 to RGBA GPU conversion", hr); return false; }
        ++decodeFrameIndex;
        return true;
    }

    bool WriteDlssOutput(LONGLONG timestamp, std::string& error) {
        D3D11_TEXTURE2D_DESC encodeDesc{};
        encodeDesc.Width = width; encodeDesc.Height = height;
        encodeDesc.MipLevels = 1; encodeDesc.ArraySize = 1;
        encodeDesc.Format = DXGI_FORMAT_NV12; encodeDesc.SampleDesc.Count = 1;
        encodeDesc.Usage = D3D11_USAGE_DEFAULT;
        encodeDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
        ID3D11Texture2D* frameTexture = nullptr;
        HRESULT hr = d3dDevice->CreateTexture2D(&encodeDesc, nullptr, &frameTexture);
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC outputViewDesc{};
        outputViewDesc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        ID3D11VideoProcessorOutputView* frameView = nullptr;
        if (SUCCEEDED(hr)) hr = videoDevice->CreateVideoProcessorOutputView(
            frameTexture, enumerator, &outputViewDesc, &frameView);
        if (FAILED(hr)) {
            Release(frameView); Release(frameTexture);
            error = HrText("Allocate NV12 encoder surface", hr); return false;
        }
        D3D11_VIDEO_PROCESSOR_STREAM stream{};
        ConfigureVideoProcessor(true);
        stream.Enable = TRUE;
        stream.InputFrameOrField = encodeFrameIndex;
        stream.pInputSurface = dlssOutputView;
        hr = videoContext->VideoProcessorBlt(
            processor, frameView, encodeFrameIndex, 1, &stream);
        Release(frameView);
        if (FAILED(hr)) {
            Release(frameTexture);
            error = HrText("RGBA to NV12 GPU conversion", hr); return false;
        }
        // Submit the RGBA->NV12 blit before the DXGI surface is transferred
        // to Media Foundation.  Without this boundary the asynchronous NVENC
        // transform can occasionally read the recycled surface too early.
        d3dContext->Flush();
        ++encodeFrameIndex;
        IMFMediaBuffer* buffer = nullptr;
        IMFSample* sample = nullptr;
        hr = MFCreateDXGISurfaceBuffer(IID_ID3D11Texture2D,
            frameTexture, 0, FALSE, &buffer);
        Release(frameTexture);
        if (FAILED(hr)) { error = HrText("Wrap NV12 DXGI surface", hr); return false; }
        hr = buffer->SetCurrentLength(width * height * 3 / 2);
        if (FAILED(hr)) { error = HrText("Set NV12 surface length", hr); Release(buffer); return false; }
        hr = MFCreateSample(&sample);
        if (SUCCEEDED(hr)) hr = sample->AddBuffer(buffer);
        const LONGLONG time = timestamp >= 0 ? timestamp : FrameTime(timelineFrameIndex);
        const LONGLONG duration = timestamp >= 0 ? frameDuration :
            std::max<LONGLONG>(1, FrameTime(timelineFrameIndex + 1) - time);
        if (SUCCEEDED(hr)) hr = sample->SetSampleTime(time);
        if (SUCCEEDED(hr)) hr = sample->SetSampleDuration(duration);
        if (SUCCEEDED(hr)) hr = writer->WriteSample(sinkStream, sample);
        ++timelineFrameIndex;
        Release(sample); Release(buffer);
        if (FAILED(hr)) { error = HrText("Submit NV12 surface to hardware encoder", hr); return false; }
        return true;
    }
};

double ParseDouble(const wchar_t* value) { return std::wcstod(value, nullptr); }
int ParseInt(const wchar_t* value) { return static_cast<int>(std::wcstol(value, nullptr, 10)); }

} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 22 && argc != 29 && argc != 31) {
        std::fprintf(stderr, "video_worker_gpu: expected 21, 28 or 30 arguments\n");
        return 2;
    }
    const std::wstring input = argv[2], output = argv[3], runtime = argv[4], core = argv[5];
    const UINT32 width = static_cast<UINT32>(ParseInt(argv[6]));
    const UINT32 height = static_cast<UINT32>(ParseInt(argv[7]));
    const double fps = std::max(0.01, ParseDouble(argv[8]));
    const UINT32 fpsNum = argc >= 31
        ? static_cast<UINT32>(std::max(1, ParseInt(argv[29])))
        : static_cast<UINT32>(std::lround(fps * 1000.0));
    const UINT32 fpsDen = argc >= 31
        ? static_cast<UINT32>(std::max(1, ParseInt(argv[30]))) : 1000u;
    const int gpu = ParseInt(argv[9]), preset = ParseInt(argv[10]);
    DlssnrSettingsC settings{};
    settings.style = ParseInt(argv[11]);
    settings.intensity = static_cast<float>(ParseDouble(argv[12]));
    settings.localTone = static_cast<float>(ParseDouble(argv[13]));
    settings.localStructure = static_cast<float>(ParseDouble(argv[14]));
    settings.skinStructure = static_cast<float>(ParseDouble(argv[15]));
    settings.autoMask = ParseInt(argv[16]); settings.uiCorrection = ParseInt(argv[17]);
    settings.paperWhiteScale = static_cast<float>(ParseDouble(argv[18]));
    settings.depthInverted = 1; settings.motionScaleX = 1.0f; settings.motionScaleY = 1.0f;
    const std::filesystem::path previewPath = argv[19];
    const std::filesystem::path stopSignal = argv[20];
    const double speedMultiplier = std::max(0.0, ParseDouble(argv[21]));
    const int guidance = argc >= 29 ? ParseInt(argv[22]) : 1;
    const float motionStrength = argc >= 29 ? static_cast<float>(ParseDouble(argv[23])) : 1.0f;
    const float depthStrength = argc >= 29 ? static_cast<float>(ParseDouble(argv[24])) : 1.0f;
    const bool reduceGrayTone = argc >= 29 && ParseInt(argv[25]) != 0;
    settings.depthInverted = argc >= 29 ? ParseInt(argv[26]) : 1;
    settings.motionScaleX = argc >= 29 ? static_cast<float>(ParseDouble(argv[27])) : 1.0f;
    settings.motionScaleY = argc >= 29 ? static_cast<float>(ParseDouble(argv[28])) : 1.0f;
    const bool useMotion = guidance == 0 || guidance == 2;
    const bool useDepth = guidance == 0 || guidance == 3;
    std::string error;
    Session session;
    if (!session.Initialize(runtime.c_str(), core.c_str(), width, height, gpu, preset, error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 3;
    }
    GpuVideoPipeline pipeline;
    if (!pipeline.Initialize(session, input.c_str(), output.c_str(), width, height,
                             fpsNum, fpsDen, error)) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    GpuColorCorrector colorCorrector;
    if ((reduceGrayTone && !colorCorrector.Initialize(session, error)) ||
        !pipeline.SetEncodeInput(
            reduceGrayTone ? colorCorrector.texture : session.output11, error)) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    GpuMotionEstimator motionEstimator;
    if (useMotion && !motionEstimator.Initialize(session, error)) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    GpuDepthEstimator depthEstimator;
    if (useDepth && !depthEstimator.Initialize(session, error)) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    uint64_t frames = 0;
    const auto started = std::chrono::steady_clock::now();
    LONGLONG lastTimestamp = -1;
    while (true) {
        if (!stopSignal.empty() && std::filesystem::exists(stopSignal)) break;
        ID3D11Texture2D* decoded = nullptr;
        UINT subresource = 0; LONGLONG timestamp = -1; bool eof = false;
        IMFSample* sample = nullptr;
        if (!pipeline.ReadTexture(&decoded, &subresource, &timestamp, &sample, eof, error)) break;
        if (eof) break;
        if (timestamp >= 0 && lastTimestamp >= 0 && timestamp <= lastTimestamp) {
            // A repeated or backwards timestamp usually means the source
            // reader re-delivered/recycled a sample while the pipeline was
            // busy.  Skipping it keeps the output timeline monotonic instead
            // of inserting a flash or a held frame.
            ++pipeline.outOfOrderSamples;
            if (sample) sample->Release();
            Release(decoded);
            continue;
        }
        lastTimestamp = timestamp;
        if (!pipeline.Nv12ToDlssInput(decoded, subresource, error)) {
            if (sample) sample->Release();
            Release(decoded); break;
        }
        if (useMotion) motionEstimator.Run(session, motionStrength);
        if (useDepth) depthEstimator.Run(session, depthStrength);
        settings.reset = frames == 0 ? 1 : 0;
        if (!session.ProcessGpuTexture(settings, error)) {
            if (sample) sample->Release();
            Release(decoded); break;
        }
        // ProcessGpuTexture guaranteed (via shared fences) that the video
        // processor has already consumed the decoded surface, so the decoder
        // sample can now be returned to its surface pool safely.
        if (sample) sample->Release();
        Release(decoded);
        if (reduceGrayTone) {
            if (frames == 0) colorCorrector.UpdateStatistics(session);
            colorCorrector.Apply(session);
        }
        // Preserve the source sample timestamps: the output stays a VFR
        // timeline identical to the container, which is what players expect
        // for screen recordings whose PTS is genuinely irregular.  Fall back
        // to a strict CFR index timeline only when the source has no PTS.
        if (!pipeline.WriteDlssOutput(timestamp, error)) break;
        ++frames;
        if (frames == 1 || frames % 15 == 0) {
            WriteGpuPreview(session,
                reduceGrayTone ? colorCorrector.texture : session.output11,
                previewPath, width, height);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::printf("PROGRESS %llu %.3f ZERO_COPY_GPU\n",
                static_cast<unsigned long long>(frames), frames / std::max(seconds, 1e-6));
            std::fflush(stdout);
        }
        if (speedMultiplier > 0.0) {
            const auto target = started + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(std::chrono::duration<double>(
                    frames / (fps * speedMultiplier)));
            std::this_thread::sleep_until(target);
        }
    }
    if (!error.empty()) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    if (!pipeline.Finish(error)) {
        std::fprintf(stderr, "%s\n", error.c_str()); return 7;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::printf("DONE %llu %.3f ZERO_COPY_GPU\n", static_cast<unsigned long long>(frames),
                frames / std::max(seconds, 1e-6));
    std::printf("STATS dataloss=%llu outoforder=%llu\n",
                static_cast<unsigned long long>(pipeline.dataLossSamples),
                static_cast<unsigned long long>(pipeline.outOfOrderSamples));
    std::fflush(stdout);
    pipeline.Destroy();
    MFShutdown();
    return frames ? 0 : 6;
}
