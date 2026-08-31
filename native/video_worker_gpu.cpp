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
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <process.h>
#include <string>
#include <thread>

#include "dlssnr_bridge.cpp"

namespace {

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
                    const double yv = (pixel[0] * c[0] + pixel[1] * c[1] + pixel[2] * c[2]) / 255.0;
                    mean[side] += yv; square[side] += yv * yv;
                    for (int channel = 0; channel < 3; ++channel) {
                        const double cv = pixel[channel] / 255.0 - yv;
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
    LONGLONG nextTime = 0;
    LONGLONG frameDuration = 166667;

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
        Release(videoContext); Release(videoDevice);
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
        if (SUCCEEDED(hr)) hr = MFCreateSinkWriterFromURL(output, nullptr, attributes, &writer);
        Release(attributes);
        if (FAILED(hr)) { error = HrText("Create hardware sink writer", hr); return false; }
        IMFMediaType* encoded = nullptr;
        hr = MFCreateMediaType(&encoded);
        if (SUCCEEDED(hr)) hr = encoded->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        if (SUCCEEDED(hr)) hr = encoded->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        if (SUCCEEDED(hr)) hr = encoded->SetUINT32(MF_MT_AVG_BITRATE,
            std::max<UINT32>(8000000, static_cast<UINT32>(width * height * fpsNum / fpsDen * 0.10)));
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
                     LONGLONG* timestamp, bool& eof, std::string& error) {
        *texture = nullptr; *subresource = 0; eof = false;
        for (;;) {
            DWORD actual = 0, flags = 0;
            IMFSample* sample = nullptr;
            HRESULT hr = reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0,
                &actual, &flags, timestamp, &sample);
            if (FAILED(hr)) { error = HrText("Read hardware decoded frame", hr); return false; }
            if (flags & MF_SOURCE_READERF_ENDOFSTREAM) { Release(sample); eof = true; return true; }
            if (!sample) continue;
            IMFMediaBuffer* buffer = nullptr;
            hr = sample->GetBufferByIndex(0, &buffer);
            IMFDXGIBuffer* dxgi = nullptr;
            if (SUCCEEDED(hr)) hr = buffer->QueryInterface(IID_IMFDXGIBuffer,
                reinterpret_cast<void**>(&dxgi));
            if (SUCCEEDED(hr)) hr = dxgi->GetResource(IID_ID3D11Texture2D,
                reinterpret_cast<void**>(texture));
            if (SUCCEEDED(hr)) hr = dxgi->GetSubresourceIndex(subresource);
            Release(dxgi); Release(buffer); Release(sample);
            if (FAILED(hr)) {
                error = "Decoder returned a CPU frame; zero-copy NV12 surface unavailable";
                return false;
            }
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
        stream.Enable = TRUE; stream.pInputSurface = view;
        if (SUCCEEDED(hr)) hr = videoContext->VideoProcessorBlt(
            processor, dlssInputView, 0, 1, &stream);
        Release(view);
        if (FAILED(hr)) { error = HrText("NV12 to RGBA GPU conversion", hr); return false; }
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
        stream.Enable = TRUE; stream.pInputSurface = dlssOutputView;
        hr = videoContext->VideoProcessorBlt(processor, frameView, 0, 1, &stream);
        Release(frameView);
        if (FAILED(hr)) {
            Release(frameTexture);
            error = HrText("RGBA to NV12 GPU conversion", hr); return false;
        }
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
        const LONGLONG time = timestamp >= 0 ? timestamp : nextTime;
        if (SUCCEEDED(hr)) hr = sample->SetSampleTime(time);
        if (SUCCEEDED(hr)) hr = sample->SetSampleDuration(frameDuration);
        if (SUCCEEDED(hr)) hr = writer->WriteSample(sinkStream, sample);
        nextTime = time + frameDuration;
        Release(sample); Release(buffer);
        if (FAILED(hr)) { error = HrText("Submit NV12 surface to hardware encoder", hr); return false; }
        return true;
    }
};

double ParseDouble(const wchar_t* value) { return std::wcstod(value, nullptr); }
int ParseInt(const wchar_t* value) { return static_cast<int>(std::wcstol(value, nullptr, 10)); }

} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 22) {
        std::fprintf(stderr, "video_worker_gpu: expected 21 arguments\n");
        return 2;
    }
    const std::wstring input = argv[2], output = argv[3], runtime = argv[4], core = argv[5];
    const UINT32 width = static_cast<UINT32>(ParseInt(argv[6]));
    const UINT32 height = static_cast<UINT32>(ParseInt(argv[7]));
    const double fps = std::max(0.01, ParseDouble(argv[8]));
    const UINT32 fpsNum = static_cast<UINT32>(std::lround(fps * 1000.0));
    constexpr UINT32 fpsDen = 1000;
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
    if (!colorCorrector.Initialize(session, error) ||
        !pipeline.SetEncodeInput(colorCorrector.texture, error)) {
        pipeline.Destroy(); session.Destroy(); MFShutdown();
        return RunCpuFallback(argc, argv, error);
    }
    uint64_t frames = 0;
    const auto started = std::chrono::steady_clock::now();
    while (true) {
        if (!stopSignal.empty() && std::filesystem::exists(stopSignal)) break;
        ID3D11Texture2D* decoded = nullptr;
        UINT subresource = 0; LONGLONG timestamp = -1; bool eof = false;
        if (!pipeline.ReadTexture(&decoded, &subresource, &timestamp, eof, error)) break;
        if (eof) break;
        if (!pipeline.Nv12ToDlssInput(decoded, subresource, error)) { Release(decoded); break; }
        settings.reset = frames == 0 ? 1 : 0;
        if (!session.ProcessGpuTexture(settings, error)) { Release(decoded); break; }
        Release(decoded);
        if (frames == 0 || frames % 15 == 0)
            colorCorrector.UpdateStatistics(session);
        colorCorrector.Apply(session);
        if (!pipeline.WriteDlssOutput(timestamp, error)) break;
        ++frames;
        if (frames == 1 || frames % 15 == 0) {
            WriteGpuPreview(session, colorCorrector.texture, previewPath, width, height);
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
    pipeline.Destroy();
    MFShutdown();
    return frames ? 0 : 6;
}
