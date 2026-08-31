#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "nvsdk_ngx.h"

namespace {

constexpr NVSDK_NGX_Feature kFeatureDlssnr = static_cast<NVSDK_NGX_Feature>(18);
constexpr unsigned long long kSnippetApplicationId = 0x0876232Cull;

template<class T> void SafeRelease(T*& value) {
    if (value) { value->Release(); value = nullptr; }
}

struct DlssnrSettingsC {
    int style;
    float intensity;
    float localTone;
    float localStructure;
    int autoMask;
    int reset;
    float skinStructure;
    int uiCorrection;
    int depthInverted;
    float motionScaleX;
    float motionScaleY;
    float paperWhiteScale;
    int colorTransfer;
};

uint16_t FloatToHalf(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000u) >> (1 - exponent);
        return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
        ((mantissa + 0x1000u) >> 13));
}

float HalfToFloat(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & 0x8000u)) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) { mantissa <<= 1; --exponent; }
            mantissa &= 0x3ffu;
            bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float SrgbToLinear(float value) {
    return value <= 0.04045f ? value / 12.92f :
        std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float value) {
    value = std::max(value, 0.0f);
    return value <= 0.0031308f ? value * 12.92f :
        1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

using CoreInitFn = NVSDK_NGX_Result (NVSDK_CONV*)(
    const char*, NVSDK_NGX_EngineType, const char*, const wchar_t*, ID3D12Device*,
    NVSDK_NGX_Version, const void*);
using AllocateParametersFn = NVSDK_NGX_Result (NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using DestroyParametersFn = NVSDK_NGX_Result (NVSDK_CONV*)(NVSDK_NGX_Parameter*);
// nvngx.dll exposes the driver ABI used by the official static wrapper.  The
// second argument receives the implementation result; the public one-argument
// API is the wrapper ABI and must not be called through GetProcAddress.
using CoreShutdownFn = NVSDK_NGX_Result (NVSDK_CONV*)(ID3D12Device*, int*);
using SnippetInitFn = NVSDK_NGX_Result (NVSDK_CONV*)(
    unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using CreateFeatureFn = NVSDK_NGX_Result (NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*,
    NVSDK_NGX_Handle**);
using EvaluateFeatureFn = NVSDK_NGX_Result (NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ReleaseFeatureFn = NVSDK_NGX_Result (NVSDK_CONV*)(NVSDK_NGX_Handle*);
using SnippetShutdownFn = NVSDK_NGX_Result (NVSDK_CONV*)(ID3D12Device*);

template<class T> T Export(HMODULE module, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(module, name));
}

void WriteError(char* output, int capacity, const std::string& value) {
    if (!output || capacity <= 0) return;
    const size_t count = std::min(value.size(), static_cast<size_t>(capacity - 1));
    std::memcpy(output, value.data(), count);
    output[count] = 0;
}

std::string HResultText(const char* stage, HRESULT hr) {
    char buffer[160]{};
    std::snprintf(buffer, sizeof(buffer), "%s failed (HRESULT=0x%08lx)", stage,
        static_cast<unsigned long>(hr));
    return buffer;
}

std::string NgxText(const char* stage, NVSDK_NGX_Result result) {
    char buffer[160]{};
    std::snprintf(buffer, sizeof(buffer), "%s failed (NGX=0x%08x)", stage,
        static_cast<unsigned int>(result));
    return buffer;
}

bool NgxOk(NVSDK_NGX_Result result) { return NVSDK_NGX_SUCCEED(result); }

// The driver-created parameter object uses NVIDIA's stable wrapper vtable
// order, which differs from the declaration order in recent public headers.
// Call the verified slots directly so this MinGW-built bridge remains ABI-safe.
template<size_t Slot, class T>
void ParamSetRaw(NVSDK_NGX_Parameter* p, const char* name, T value) {
    using Fn = void (NVSDK_CONV*)(NVSDK_NGX_Parameter*, const char*, T);
    auto table = *reinterpret_cast<void***>(p);
    reinterpret_cast<Fn>(table[Slot])(p, name, value);
}
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, void* v) { ParamSetRaw<0>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, ID3D12Resource* v) { ParamSetRaw<1>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, ID3D11Resource* v) { ParamSetRaw<2>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, int v) { ParamSetRaw<3>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, unsigned int v) { ParamSetRaw<4>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, double v) { ParamSetRaw<5>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, float v) { ParamSetRaw<6>(p, n, v); }
void ParamSet(NVSDK_NGX_Parameter* p, const char* n, unsigned long long v) { ParamSetRaw<7>(p, n, v); }

std::atomic<HMODULE> gCallerModule{nullptr};
std::atomic<void*> gOriginalGetModuleFileName{nullptr};
std::atomic<void*> gHookOwner{nullptr};

DWORD WINAPI HookGetModuleFileNameW(HMODULE module, LPWSTR filename, DWORD size) {
    if (module == gCallerModule.load(std::memory_order_acquire)) {
        constexpr wchar_t name[] = L"nvngx.dll";
        constexpr DWORD length = static_cast<DWORD>(std::size(name) - 1);
        if (!filename || size == 0) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return 0; }
        if (size <= length) {
            if (size > 1) std::memcpy(filename, name, (size - 1) * sizeof(wchar_t));
            filename[size - 1] = 0;
            SetLastError(ERROR_INSUFFICIENT_BUFFER);
            return size;
        }
        std::memcpy(filename, name, sizeof(name));
        return length;
    }
    using Fn = DWORD (WINAPI*)(HMODULE, LPWSTR, DWORD);
    auto fn = reinterpret_cast<Fn>(gOriginalGetModuleFileName.load(std::memory_order_acquire));
    return fn ? fn(module, filename, size) : 0;
}

void** FindImportedFunctionSlot(HMODULE module, const char* functionName) {
    if (!module || !functionName) return nullptr;
    auto* base = reinterpret_cast<std::byte*>(module);
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return nullptr;
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress || !dir.Size) return nullptr;
    auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
    auto* end = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress + dir.Size);
    for (; descriptor < end && descriptor->Name; ++descriptor) {
        const char* library = reinterpret_cast<const char*>(base + descriptor->Name);
        if (_stricmp(library, "KERNEL32.dll") != 0 &&
            _stricmp(library, "api-ms-win-core-libraryloader-l1-2-0.dll") != 0 &&
            _stricmp(library, "api-ms-win-core-libraryloader-l1-1-0.dll") != 0) continue;
        if (!descriptor->OriginalFirstThunk || !descriptor->FirstThunk) continue;
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + descriptor->OriginalFirstThunk);
        auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + descriptor->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++addresses) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            auto* imported = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(
                base + static_cast<uint32_t>(names->u1.AddressOfData));
            if (std::strcmp(reinterpret_cast<const char*>(imported->Name), functionName) == 0)
                return reinterpret_cast<void**>(&addresses->u1.Function);
        }
    }
    return nullptr;
}

struct Session {
    std::mutex mutex;
    uint32_t width = 0;
    uint32_t height = 0;
    std::wstring runtimePath;
    std::wstring runtimeDirectory;
    std::wstring cacheDirectory;
    HMODULE coreModule = nullptr;
    HMODULE snippetModule = nullptr;
    void** iatSlot = nullptr;
    void* originalIat = nullptr;
    bool hookInstalled = false;
    bool coreInitialized = false;
    bool snippetInitialized = false;
    bool historyStarted = false;
    bool zeroGuidanceInitialized = false;
    uint64_t fenceValue = 0;

    IDXGIFactory6* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D11Device* device11Base = nullptr;
    ID3D11Device5* device11 = nullptr;
    ID3D11DeviceContext* context11Base = nullptr;
    ID3D11DeviceContext4* context11 = nullptr;
    ID3D12Device* device12 = nullptr;
    ID3D12CommandQueue* queue12 = nullptr;
    ID3D12CommandAllocator* allocator12 = nullptr;
    ID3D12GraphicsCommandList* list12 = nullptr;
    ID3D11Fence* fence11 = nullptr;
    ID3D12Fence* fence12 = nullptr;
    HANDLE fenceEvent = nullptr;

    ID3D11Texture2D* input11 = nullptr;
    ID3D11Texture2D* output11 = nullptr;
    ID3D11Texture2D* motion11 = nullptr;
    ID3D11Texture2D* depth11 = nullptr;
    ID3D11Texture2D* staging11 = nullptr;
    ID3D12Resource* input12 = nullptr;
    ID3D12Resource* output12 = nullptr;
    ID3D12Resource* motion12 = nullptr;
    ID3D12Resource* depth12 = nullptr;
    std::vector<uint8_t> colorUpload;

    CoreInitFn coreInit = nullptr;
    AllocateParametersFn allocateParameters = nullptr;
    DestroyParametersFn destroyParameters = nullptr;
    CoreShutdownFn coreShutdown = nullptr;
    SnippetInitFn snippetInit = nullptr;
    CreateFeatureFn createFeature = nullptr;
    EvaluateFeatureFn evaluateFeature = nullptr;
    ReleaseFeatureFn releaseFeature = nullptr;
    SnippetShutdownFn snippetShutdown = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;

    ~Session() { Destroy(); }

    bool InstallHook(std::string& error) {
        iatSlot = FindImportedFunctionSlot(snippetModule, "GetModuleFileNameW");
        if (!iatSlot) { error = "DLSSNR runtime has no GetModuleFileNameW import"; return false; }
        void* expected = nullptr;
        if (!gHookOwner.compare_exchange_strong(expected, this)) {
            error = "Another DLSSNR live session is already active"; return false;
        }
        HMODULE caller = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&HookGetModuleFileNameW), &caller)) {
            gHookOwner.store(nullptr); error = "Cannot resolve bridge module"; return false;
        }
        DWORD oldProtect = 0;
        if (!VirtualProtect(iatSlot, sizeof(void*), PAGE_READWRITE, &oldProtect)) {
            gHookOwner.store(nullptr); error = "Cannot patch DLSSNR caller compatibility"; return false;
        }
        gCallerModule.store(caller);
        originalIat = InterlockedExchangePointer(reinterpret_cast<void* volatile*>(iatSlot),
            reinterpret_cast<void*>(&HookGetModuleFileNameW));
        gOriginalGetModuleFileName.store(originalIat);
        DWORD ignored = 0; VirtualProtect(iatSlot, sizeof(void*), oldProtect, &ignored);
        FlushInstructionCache(GetCurrentProcess(), iatSlot, sizeof(void*));
        hookInstalled = true;
        return true;
    }

    void RestoreHook() {
        if (!hookInstalled) return;
        DWORD oldProtect = 0;
        if (iatSlot && VirtualProtect(iatSlot, sizeof(void*), PAGE_READWRITE, &oldProtect)) {
            InterlockedExchangePointer(reinterpret_cast<void* volatile*>(iatSlot), originalIat);
            DWORD ignored = 0; VirtualProtect(iatSlot, sizeof(void*), oldProtect, &ignored);
            FlushInstructionCache(GetCurrentProcess(), iatSlot, sizeof(void*));
        }
        hookInstalled = false;
        gOriginalGetModuleFileName.store(nullptr);
        gCallerModule.store(nullptr);
        gHookOwner.store(nullptr);
    }

    bool CreateSharedTexture(DXGI_FORMAT format, UINT bindFlags,
        ID3D11Texture2D** out11, ID3D12Resource** out12, std::string& error) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1;
        desc.Format = format; desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = bindFlags;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        HRESULT hr = device11->CreateTexture2D(&desc, nullptr, out11);
        if (FAILED(hr)) { error = HResultText("Create shared D3D11 texture", hr); return false; }
        IDXGIResource1* resource = nullptr;
        hr = (*out11)->QueryInterface(IID_IDXGIResource1, reinterpret_cast<void**>(&resource));
        if (FAILED(hr)) { error = HResultText("Query IDXGIResource1", hr); return false; }
        HANDLE shared = nullptr;
        hr = resource->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &shared);
        resource->Release();
        if (FAILED(hr)) { error = HResultText("Create shared texture handle", hr); return false; }
        hr = device12->OpenSharedHandle(shared, IID_ID3D12Resource,
            reinterpret_cast<void**>(out12));
        CloseHandle(shared);
        if (FAILED(hr)) { error = HResultText("Open shared texture in D3D12", hr); return false; }
        return true;
    }

    bool Wait(uint64_t value, std::string& error) {
        if (fence12->GetCompletedValue() >= value) return true;
        HRESULT hr = fence12->SetEventOnCompletion(value, fenceEvent);
        if (FAILED(hr)) { error = HResultText("Set fence event", hr); return false; }
        if (WaitForSingleObject(fenceEvent, 60000) != WAIT_OBJECT_0) {
            error = "DLSSNR GPU wait timed out"; return false;
        }
        return true;
    }

    void SetSubrect(const char* baseX, const char* baseY,
        const char* rectWidth, const char* rectHeight) {
        ParamSet(parameters, baseX, 0u);
        ParamSet(parameters, baseY, 0u);
        ParamSet(parameters, rectWidth, width);
        ParamSet(parameters, rectHeight, height);
    }

    static NVSDK_NGX_Result NVSDK_CONV ScalingRatioCallback(NVSDK_NGX_Parameter* p) {
        if (!p) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        ParamSet(p, "DLSSNR.ScalingRatio", 1.0f);
        return NVSDK_NGX_Result_Success;
    }

    bool Initialize(const wchar_t* runtime, const wchar_t* core, uint32_t w,
        uint32_t h, int requestedGpu, int requestedPreset, std::string& error) {
        width = w; height = h; runtimePath = runtime;
        runtimeDirectory = std::filesystem::path(runtimePath).parent_path().wstring();
        if (!width || !height || !std::filesystem::exists(runtimePath)) {
            error = "Invalid DLSSNR runtime path or image dimensions"; return false;
        }
        coreModule = LoadLibraryExW(core, nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!coreModule) { error = "Cannot load NVIDIA NGX core DLL"; return false; }
        coreInit = Export<CoreInitFn>(coreModule, "NVSDK_NGX_D3D12_Init_ProjectID");
        allocateParameters = Export<AllocateParametersFn>(coreModule, "NVSDK_NGX_D3D12_AllocateParameters");
        destroyParameters = Export<DestroyParametersFn>(coreModule, "NVSDK_NGX_D3D12_DestroyParameters");
        coreShutdown = Export<CoreShutdownFn>(coreModule, "NVSDK_NGX_D3D12_Shutdown1");
        if (!coreInit || !allocateParameters || !destroyParameters || !coreShutdown) {
            error = "NVIDIA NGX core exports are incomplete"; return false;
        }

        HRESULT hr = CreateDXGIFactory1(IID_IDXGIFactory6, reinterpret_cast<void**>(&factory));
        if (FAILED(hr)) { error = HResultText("Create DXGI factory", hr); return false; }
        int nvidiaIndex = 0;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{}; adapter->GetDesc1(&desc);
            if (desc.VendorId == 0x10DE && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                if (nvidiaIndex == std::max(0, requestedGpu)) break;
                ++nvidiaIndex;
            }
            adapter->Release(); adapter = nullptr;
        }
        if (!adapter) { error = "Requested NVIDIA DXGI adapter was not found"; return false; }
        hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
            D3D11_SDK_VERSION, &device11Base, nullptr, &context11Base);
        if (FAILED(hr)) { error = HResultText("Create D3D11 device", hr); return false; }
        hr = device11Base->QueryInterface(IID_ID3D11Device5, reinterpret_cast<void**>(&device11));
        if (SUCCEEDED(hr)) hr = context11Base->QueryInterface(IID_ID3D11DeviceContext4,
            reinterpret_cast<void**>(&context11));
        if (FAILED(hr)) { error = HResultText("Query D3D11.4 interfaces", hr); return false; }
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_ID3D12Device,
            reinterpret_cast<void**>(&device12));
        if (FAILED(hr)) { error = HResultText("Create D3D12 device", hr); return false; }
        D3D12_COMMAND_QUEUE_DESC queueDesc{}; queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = device12->CreateCommandQueue(&queueDesc, IID_ID3D12CommandQueue,
            reinterpret_cast<void**>(&queue12));
        if (SUCCEEDED(hr)) hr = device12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_ID3D12CommandAllocator, reinterpret_cast<void**>(&allocator12));
        if (SUCCEEDED(hr)) hr = device12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator12, nullptr, IID_ID3D12GraphicsCommandList, reinterpret_cast<void**>(&list12));
        if (FAILED(hr)) { error = HResultText("Create D3D12 command objects", hr); return false; }

        const UINT colorBind = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS |
            D3D11_BIND_RENDER_TARGET;
        // DLSSNR's SDR path consumes RGBA8.  Keeping color in this format
        // matches the real-time renderer and avoids two full-frame FP16 CPU
        // conversions on every image/video frame.
        if (!CreateSharedTexture(DXGI_FORMAT_R8G8B8A8_UNORM, colorBind, &input11, &input12, error) ||
            !CreateSharedTexture(DXGI_FORMAT_R8G8B8A8_UNORM, colorBind, &output11, &output12, error) ||
            !CreateSharedTexture(DXGI_FORMAT_R16G16_FLOAT, colorBind,
                &motion11, &motion12, error) ||
            !CreateSharedTexture(DXGI_FORMAT_R32_FLOAT, colorBind,
                &depth11, &depth12, error)) return false;
        D3D11_TEXTURE2D_DESC stagingDesc{}; output11->GetDesc(&stagingDesc);
        stagingDesc.Usage = D3D11_USAGE_STAGING; stagingDesc.BindFlags = 0;
        stagingDesc.MiscFlags = 0; stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        hr = device11->CreateTexture2D(&stagingDesc, nullptr, &staging11);
        if (FAILED(hr)) { error = HResultText("Create readback texture", hr); return false; }

        hr = device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_ID3D11Fence,
            reinterpret_cast<void**>(&fence11));
        HANDLE fenceHandle = nullptr;
        if (SUCCEEDED(hr)) hr = fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fenceHandle);
        if (SUCCEEDED(hr)) hr = device12->OpenSharedHandle(fenceHandle, IID_ID3D12Fence,
            reinterpret_cast<void**>(&fence12));
        if (fenceHandle) CloseHandle(fenceHandle);
        fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (FAILED(hr) || !fenceEvent) { error = HResultText("Create shared fence", hr); return false; }

        wchar_t localAppData[MAX_PATH]{};
        DWORD localAppDataLength = GetEnvironmentVariableW(
            L"LOCALAPPDATA", localAppData, static_cast<DWORD>(std::size(localAppData)));
        const std::filesystem::path cacheBase = localAppDataLength &&
            localAppDataLength < std::size(localAppData)
            ? std::filesystem::path(localAppData)
            : std::filesystem::temp_directory_path();
        cacheDirectory = (cacheBase / L"ComfyUI-DLSSNR-Live" / L"NGXCache").wstring();
        std::error_code cacheError;
        std::filesystem::create_directories(cacheDirectory, cacheError);
        if (cacheError) { error = "Cannot create isolated NGX cache directory"; return false; }
        const wchar_t* featurePaths[]{ runtimeDirectory.c_str() };
        NVSDK_NGX_FeatureCommonInfo featureInfo{};
        featureInfo.PathListInfo.Path = featurePaths;
        featureInfo.PathListInfo.Length = 1;
        auto ngx = coreInit(
            "7c134ab9-9677-4af5-a2b2-bca943350861",
            NVSDK_NGX_ENGINE_TYPE_CUSTOM, "ComfyUI-DLSSNR-Live-1",
            cacheDirectory.c_str(), device12, NVSDK_NGX_Version_API, &featureInfo);
        if (!NgxOk(ngx)) { error = NgxText("NGX core init", ngx); return false; }
        coreInitialized = true;

        snippetModule = LoadLibraryExW(runtimePath.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!snippetModule) { error = "Cannot load nvngx_dlssnr.dll"; return false; }
        snippetInit = Export<SnippetInitFn>(snippetModule, "NVSDK_NGX_D3D12_Init_Ext");
        createFeature = Export<CreateFeatureFn>(snippetModule, "NVSDK_NGX_D3D12_CreateFeature");
        evaluateFeature = Export<EvaluateFeatureFn>(snippetModule, "NVSDK_NGX_D3D12_EvaluateFeature");
        releaseFeature = Export<ReleaseFeatureFn>(snippetModule, "NVSDK_NGX_D3D12_ReleaseFeature");
        snippetShutdown = Export<SnippetShutdownFn>(snippetModule, "NVSDK_NGX_D3D12_Shutdown1");
        if (!snippetInit || !createFeature || !evaluateFeature || !releaseFeature || !snippetShutdown) {
            error = "DLSSNR D3D12 exports are incomplete"; return false;
        }
        if (!InstallHook(error)) return false;
        ngx = snippetInit(kSnippetApplicationId, runtimeDirectory.c_str(), device12,
            NVSDK_NGX_Version_API, nullptr);
        if (!NgxOk(ngx)) { error = NgxText("DLSSNR snippet init", ngx); return false; }
        snippetInitialized = true;
        ngx = allocateParameters(&parameters);
        if (!NgxOk(ngx) || !parameters) { error = NgxText("Allocate NGX parameters", ngx); return false; }

        ParamSet(parameters, "DLSSNR.Width", width); ParamSet(parameters, "DLSSNR.Height", height);
        ParamSet(parameters, "DLSSNR.InputWidth", width); ParamSet(parameters, "DLSSNR.InputHeight", height);
        ParamSet(parameters, "DLSSNR.OutputWidth", width); ParamSet(parameters, "DLSSNR.OutputHeight", height);
        ParamSet(parameters, "DLSSNR.Output.Width", width); ParamSet(parameters, "DLSSNR.Output.Height", height);
        ParamSet(parameters, "DLSSNR.Upscaling", 0u); ParamSet(parameters, "DLSSNR.Scale", 1.0f);
        ParamSet(parameters, "DLSSNR.ScalingRatio", 1.0f);
        ParamSet(parameters, "DLSSNRComputeScalingRatioCallback",
            reinterpret_cast<void*>(&ScalingRatioCallback));
        ParamSet(parameters, "DLSSNR.Hint.Render.Preset",
            std::clamp(requestedPreset, 0, 3));
        ParamSet(parameters, NVSDK_NGX_Parameter_Width, width);
        ParamSet(parameters, NVSDK_NGX_Parameter_Height, height);
        ParamSet(parameters, NVSDK_NGX_Parameter_PerfQualityValue,
            static_cast<int>(NVSDK_NGX_PerfQuality_Value_Balanced));
        ParamSet(parameters, NVSDK_NGX_Parameter_CreationNodeMask, 1u);
        ParamSet(parameters, NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
        ngx = createFeature(list12, kFeatureDlssnr, parameters, &feature);
        if (!NgxOk(ngx) || !feature) { error = NgxText("Create DLSSNR Feature 18", ngx); return false; }
        hr = list12->Close();
        if (FAILED(hr)) { error = HResultText("Close initialization command list", hr); return false; }
        ID3D12CommandList* lists[]{list12}; queue12->ExecuteCommandLists(1, lists);
        const uint64_t done = ++fenceValue; queue12->Signal(fence12, done);
        return Wait(done, error);
    }

    // Evaluate a frame which has already been written into input11.  The
    // result remains in output11.  This is the zero-readback entry point used
    // by the V1.4 Media Foundation video pipeline.
    bool ProcessGpuTexture(const DlssnrSettingsC& settings, std::string& error) {
        std::lock_guard lock(mutex);
        if (!feature || !parameters || !input11 || !output11) {
            error = "Invalid GPU texture session";
            return false;
        }
        if (!zeroGuidanceInitialized) {
            std::vector<float> zeroDepth(static_cast<size_t>(width) * height);
            std::vector<uint16_t> zeroMotion(static_cast<size_t>(width) * height * 2);
            context11->UpdateSubresource(depth11, 0, nullptr, zeroDepth.data(), width * 4, 0);
            context11->UpdateSubresource(motion11, 0, nullptr, zeroMotion.data(), width * 4, 0);
            zeroGuidanceInitialized = true;
        }
        const uint64_t inputReady = ++fenceValue;
        HRESULT hr = context11->Signal(fence11, inputReady);
        context11->Flush();
        if (FAILED(hr) || FAILED(queue12->Wait(fence12, inputReady))) {
            error = HResultText("Synchronize GPU texture input", hr);
            return false;
        }
        hr = allocator12->Reset();
        if (SUCCEEDED(hr)) hr = list12->Reset(allocator12, nullptr);
        if (FAILED(hr)) { error = HResultText("Reset GPU command list", hr); return false; }
        ID3D12Resource* resources[]{input12, output12, motion12, depth12};
        D3D12_RESOURCE_BARRIER barriers[4]{};
        for (int i = 0; i < 4; ++i) {
            barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[i].Transition.pResource = resources[i];
            barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barriers[i].Transition.StateAfter = i == 1 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        list12->ResourceBarrier(4, barriers);
        ParamSet(parameters, "DLSSNR.Color", input12); ParamSet(parameters, "DLSSNR.Output", output12);
        ParamSet(parameters, "DLSSNR.MVec", motion12); ParamSet(parameters, "DLSSNR.Depth", depth12);
        SetSubrect("DLSSNR.ColorSubrectBaseX", "DLSSNR.ColorSubrectBaseY",
            "DLSSNR.ColorSubrectWidth", "DLSSNR.ColorSubrectHeight");
        SetSubrect("DLSSNR.OutputSubrectBaseX", "DLSSNR.OutputSubrectBaseY",
            "DLSSNR.OutputSubrectWidth", "DLSSNR.OutputSubrectHeight");
        SetSubrect("DLSSNR.MVecSubrectBaseX", "DLSSNR.MVecSubrectBaseY",
            "DLSSNR.MVecSubrectWidth", "DLSSNR.MVecSubrectHeight");
        SetSubrect("DLSSNR.DepthSubrectBaseX", "DLSSNR.DepthSubrectBaseY",
            "DLSSNR.DepthSubrectWidth", "DLSSNR.DepthSubrectHeight");
        ParamSet(parameters, "DLSSNR.MVecScaleX", std::clamp(settings.motionScaleX, -4.0f, 4.0f));
        ParamSet(parameters, "DLSSNR.MVecScaleY", std::clamp(settings.motionScaleY, -4.0f, 4.0f));
        ParamSet(parameters, "DLSSNR.DepthInverted", settings.depthInverted ? 1 : 0);
        ParamSet(parameters, "DLSSNR.Enabled", 1);
        ParamSet(parameters, "DLSSNR.Reset", (!historyStarted || settings.reset) ? 1 : 0);
        ParamSet(parameters, "DLSSNR.Style", std::clamp(settings.style, 0, 2));
        ParamSet(parameters, "DLSSNR.Intensity", std::clamp(settings.intensity, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.LocalToneStrength", std::clamp(settings.localTone, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.LocalStructureStrength", std::clamp(settings.localStructure, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.SkinStructureStrength", std::clamp(settings.skinStructure, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.UseAutoMask", settings.autoMask ? 1 : 0);
        ParamSet(parameters, "DLSSNR.UICorrection", settings.uiCorrection ? 1 : 0);
        ParamSet(parameters, "DLSS.Indicator.Invert.X.Axis", 0);
        ParamSet(parameters, "DLSS.Indicator.Invert.Y.Axis", 0);
        const auto ngx = evaluateFeature(list12, feature, parameters, nullptr);
        if (!NgxOk(ngx)) { error = NgxText("Evaluate DLSSNR GPU texture", ngx); list12->Close(); return false; }
        for (auto& barrier : barriers)
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        list12->ResourceBarrier(4, barriers);
        hr = list12->Close();
        if (FAILED(hr)) { error = HResultText("Close GPU evaluate list", hr); return false; }
        ID3D12CommandList* lists[]{list12}; queue12->ExecuteCommandLists(1, lists);
        const uint64_t outputReady = ++fenceValue;
        hr = queue12->Signal(fence12, outputReady);
        if (SUCCEEDED(hr)) hr = context11->Wait(fence11, outputReady);
        if (FAILED(hr)) { error = HResultText("Synchronize GPU texture output", hr); return false; }
        // The caller reuses the shared input/output textures on the next
        // frame.  A D3D11 Wait only queues a GPU-side dependency; wait for the
        // fence on the CPU here so decoder surfaces and temporal history are
        // never overwritten while NGX still consumes them.
        if (!Wait(outputReady, error)) return false;
        historyStarted = true;
        return true;
    }

    bool Process(const uint8_t* input, uint32_t inputStride, const float* depth,
        uint32_t depthStride, const uint16_t* motion, uint32_t motionStride,
        const DlssnrSettingsC& settings, uint8_t* output, uint32_t outputStride,
        std::string& error) {
        std::lock_guard lock(mutex);
        if (!input || !output || !feature || !parameters) { error = "Invalid live session"; return false; }
        const bool linearRoundtrip = settings.colorTransfer != 0;
        const uint8_t* upload = input;
        uint32_t uploadStride = inputStride;
        if (linearRoundtrip) {
            colorUpload.resize(static_cast<size_t>(width) * height * 4);
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t y = 0; y < static_cast<int64_t>(height); ++y) {
                const auto* source = input + static_cast<size_t>(y) * inputStride;
                auto* destination = colorUpload.data() + static_cast<size_t>(y) * width * 4;
                for (uint32_t x = 0; x < width; ++x) {
                    for (uint32_t channel = 0; channel < 3; ++channel) {
                        const float value = SrgbToLinear(source[x * 4 + channel] / 255.0f);
                        destination[x * 4 + channel] = static_cast<uint8_t>(
                            std::clamp(value * 255.0f + 0.5f, 0.0f, 255.0f));
                    }
                    destination[x * 4 + 3] = source[x * 4 + 3];
                }
            }
            upload = colorUpload.data();
            uploadStride = width * 4;
        }
        context11->UpdateSubresource(input11, 0, nullptr, upload, uploadStride, 0);
        std::vector<float> zeroDepth;
        std::vector<uint16_t> zeroMotion;
        if (!depth) { zeroDepth.resize(static_cast<size_t>(width) * height); depth = zeroDepth.data(); depthStride = width * 4; }
        if (!motion) { zeroMotion.resize(static_cast<size_t>(width) * height * 2); motion = zeroMotion.data(); motionStride = width * 4; }
        context11->UpdateSubresource(depth11, 0, nullptr, depth, depthStride, 0);
        context11->UpdateSubresource(motion11, 0, nullptr, motion, motionStride, 0);
        const uint64_t inputReady = ++fenceValue;
        HRESULT hr = context11->Signal(fence11, inputReady);
        context11->Flush();
        if (FAILED(hr) || FAILED(queue12->Wait(fence12, inputReady))) {
            error = HResultText("Synchronize D3D11 input", hr); return false;
        }
        hr = allocator12->Reset();
        if (SUCCEEDED(hr)) hr = list12->Reset(allocator12, nullptr);
        if (FAILED(hr)) { error = HResultText("Reset command list", hr); return false; }
        ID3D12Resource* resources[]{input12, output12, motion12, depth12};
        D3D12_RESOURCE_BARRIER barriers[4]{};
        for (int i = 0; i < 4; ++i) {
            barriers[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barriers[i].Transition.pResource = resources[i];
            barriers[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barriers[i].Transition.StateAfter = i == 1 ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        list12->ResourceBarrier(4, barriers);
        ParamSet(parameters, "DLSSNR.Color", input12); ParamSet(parameters, "DLSSNR.Output", output12);
        ParamSet(parameters, "DLSSNR.MVec", motion12); ParamSet(parameters, "DLSSNR.Depth", depth12);
        SetSubrect("DLSSNR.ColorSubrectBaseX", "DLSSNR.ColorSubrectBaseY",
            "DLSSNR.ColorSubrectWidth", "DLSSNR.ColorSubrectHeight");
        SetSubrect("DLSSNR.OutputSubrectBaseX", "DLSSNR.OutputSubrectBaseY",
            "DLSSNR.OutputSubrectWidth", "DLSSNR.OutputSubrectHeight");
        SetSubrect("DLSSNR.MVecSubrectBaseX", "DLSSNR.MVecSubrectBaseY",
            "DLSSNR.MVecSubrectWidth", "DLSSNR.MVecSubrectHeight");
        SetSubrect("DLSSNR.DepthSubrectBaseX", "DLSSNR.DepthSubrectBaseY",
            "DLSSNR.DepthSubrectWidth", "DLSSNR.DepthSubrectHeight");
        ParamSet(parameters, "DLSSNR.MVecScaleX", std::clamp(settings.motionScaleX, -4.0f, 4.0f));
        ParamSet(parameters, "DLSSNR.MVecScaleY", std::clamp(settings.motionScaleY, -4.0f, 4.0f));
        ParamSet(parameters, "DLSSNR.DepthInverted", settings.depthInverted ? 1 : 0);
        ParamSet(parameters, "DLSSNR.Enabled", 1);
        ParamSet(parameters, "DLSSNR.Reset", (!historyStarted || settings.reset) ? 1 : 0);
        ParamSet(parameters, "DLSSNR.Style", std::clamp(settings.style, 0, 2));
        ParamSet(parameters, "DLSSNR.Intensity", std::clamp(settings.intensity, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.LocalToneStrength", std::clamp(settings.localTone, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.LocalStructureStrength", std::clamp(settings.localStructure, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.SkinStructureStrength", std::clamp(settings.skinStructure, -1.0f, 2.0f));
        ParamSet(parameters, "DLSSNR.UseAutoMask", settings.autoMask ? 1 : 0);
        ParamSet(parameters, "DLSSNR.UICorrection", settings.uiCorrection ? 1 : 0);
        ParamSet(parameters, "DLSS.Indicator.Invert.X.Axis", 0);
        ParamSet(parameters, "DLSS.Indicator.Invert.Y.Axis", 0);
        auto ngx = evaluateFeature(list12, feature, parameters, nullptr);
        if (!NgxOk(ngx)) { error = NgxText("Evaluate DLSSNR", ngx); list12->Close(); return false; }
        for (auto& barrier : barriers)
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        list12->ResourceBarrier(4, barriers);
        hr = list12->Close();
        if (FAILED(hr)) { error = HResultText("Close evaluate command list", hr); return false; }
        ID3D12CommandList* lists[]{list12}; queue12->ExecuteCommandLists(1, lists);
        const uint64_t outputReady = ++fenceValue;
        hr = queue12->Signal(fence12, outputReady);
        if (SUCCEEDED(hr)) hr = context11->Wait(fence11, outputReady);
        if (FAILED(hr)) { error = HResultText("Synchronize DLSSNR output", hr); return false; }
        context11->CopyResource(staging11, output11);
        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context11->Map(staging11, 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) { error = HResultText("Map DLSSNR output", hr); return false; }
        const float paperWhite = std::clamp(settings.paperWhiteScale, 0.01f, 16.0f);
        if (!linearRoundtrip && std::abs(paperWhite - 1.0f) < 0.0001f) {
            for (uint32_t y = 0; y < height; ++y) {
                std::memcpy(output + static_cast<size_t>(y) * outputStride,
                    static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(y) * mapped.RowPitch,
                    static_cast<size_t>(width) * 4);
            }
        } else {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t y = 0; y < static_cast<int64_t>(height); ++y) {
                auto* destination = output + static_cast<size_t>(y) * outputStride;
                const auto* source = static_cast<const uint8_t*>(mapped.pData) +
                    static_cast<size_t>(y) * mapped.RowPitch;
                for (uint32_t x = 0; x < width; ++x) {
                    for (uint32_t channel = 0; channel < 3; ++channel) {
                        float value = source[x * 4 + channel] / 255.0f;
                        float adjusted = paperWhite <= 1.0f
                            ? value * paperWhite
                            : (value * paperWhite) / (1.0f + value * (paperWhite - 1.0f));
                        if (linearRoundtrip) adjusted = LinearToSrgb(adjusted);
                        destination[x * 4 + channel] = static_cast<uint8_t>(
                            std::clamp(adjusted * 255.0f + 0.5f, 0.0f, 255.0f));
                    }
                    destination[x * 4 + 3] = source[x * 4 + 3];
                }
            }
        }
        context11->Unmap(staging11, 0);
        historyStarted = true;
        return true;
    }

    void Destroy() {
        std::lock_guard lock(mutex);
        if (queue12 && fence12 && fenceEvent) {
            const uint64_t done = ++fenceValue;
            if (SUCCEEDED(queue12->Signal(fence12, done))) {
                std::string ignored; Wait(done, ignored);
            }
        }
        if (feature && releaseFeature) { releaseFeature(feature); feature = nullptr; }
        if (parameters && destroyParameters) { destroyParameters(parameters); parameters = nullptr; }
        if (snippetInitialized && snippetShutdown && device12) snippetShutdown(device12);
        snippetInitialized = false;
        RestoreHook();
        if (snippetModule) { FreeLibrary(snippetModule); snippetModule = nullptr; }
        if (coreInitialized && coreShutdown && device12) {
            int implementationResult = 0;
            coreShutdown(device12, &implementationResult);
        }
        coreInitialized = false;
        SafeRelease(staging11); SafeRelease(depth11); SafeRelease(motion11);
        SafeRelease(output11); SafeRelease(input11);
        SafeRelease(depth12); SafeRelease(motion12); SafeRelease(output12); SafeRelease(input12);
        SafeRelease(fence12); SafeRelease(fence11);
        SafeRelease(list12); SafeRelease(allocator12); SafeRelease(queue12); SafeRelease(device12);
        SafeRelease(context11); SafeRelease(context11Base); SafeRelease(device11); SafeRelease(device11Base);
        SafeRelease(adapter); SafeRelease(factory);
        if (fenceEvent) { CloseHandle(fenceEvent); fenceEvent = nullptr; }
        if (coreModule) { FreeLibrary(coreModule); coreModule = nullptr; }
    }
};

} // namespace

extern "C" __declspec(dllexport) void dlssnr_restore_color(
    const uint8_t* source, uint32_t sourceStride, uint8_t* output,
    uint32_t outputStride, uint32_t width, uint32_t height, float strength) {
    if (!source || !output || !width || !height || strength <= 0.0f) return;
    const size_t pixels = static_cast<size_t>(width) * height;
    double sourceMean = 0.0, outputMean = 0.0, sourceSq = 0.0, outputSq = 0.0;
    double sourceChroma[3]{}, outputChroma[3]{};
    constexpr double c[3]{0.2126, 0.7152, 0.0722};
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sourceMean,outputMean,sourceSq,outputSq,sourceChroma[:3],outputChroma[:3])
    #endif
    for (int64_t y = 0; y < static_cast<int64_t>(height); ++y) {
        const auto* a = source + static_cast<size_t>(y) * sourceStride;
        const auto* b = output + static_cast<size_t>(y) * outputStride;
        for (uint32_t x = 0; x < width; ++x) {
            const double ya = (a[x*4]*c[0] + a[x*4+1]*c[1] + a[x*4+2]*c[2]) / 255.0;
            const double yb = (b[x*4]*c[0] + b[x*4+1]*c[1] + b[x*4+2]*c[2]) / 255.0;
            sourceMean += ya; outputMean += yb; sourceSq += ya*ya; outputSq += yb*yb;
            for (int channel=0; channel<3; ++channel) {
                sourceChroma[channel] += a[x*4+channel]/255.0 - ya;
                outputChroma[channel] += b[x*4+channel]/255.0 - yb;
            }
        }
    }
    sourceMean /= pixels; outputMean /= pixels;
    for (int channel=0; channel<3; ++channel) {
        sourceChroma[channel] /= pixels; outputChroma[channel] /= pixels;
    }
    const double sourceStd = std::sqrt(std::max(sourceSq/pixels-sourceMean*sourceMean,1e-8));
    const double outputStd = std::sqrt(std::max(outputSq/pixels-outputMean*outputMean,1e-8));
    double sourceEnergy=0.0, outputEnergy=0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sourceEnergy,outputEnergy)
    #endif
    for (int64_t y=0; y<static_cast<int64_t>(height); ++y) {
        const auto* a=source+static_cast<size_t>(y)*sourceStride;
        const auto* b=output+static_cast<size_t>(y)*outputStride;
        for (uint32_t x=0; x<width; ++x) {
            const double ya=(a[x*4]*c[0]+a[x*4+1]*c[1]+a[x*4+2]*c[2])/255.0;
            const double yb=(b[x*4]*c[0]+b[x*4+1]*c[1]+b[x*4+2]*c[2])/255.0;
            for(int channel=0;channel<3;++channel){
                const double ca=a[x*4+channel]/255.0-ya-sourceChroma[channel];
                const double cb=b[x*4+channel]/255.0-yb-outputChroma[channel];
                sourceEnergy+=ca*ca; outputEnergy+=cb*cb;
            }
        }
    }
    const double lumaGain=std::clamp(sourceStd/outputStd,0.5,2.0);
    const double chromaGain=std::clamp(std::sqrt(std::max(sourceEnergy,1e-8)/
        std::max(outputEnergy,1e-8)),0.5,2.0);
    const double amount=std::clamp<double>(strength,0.0,1.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t y=0;y<static_cast<int64_t>(height);++y) {
        const auto* a=source+static_cast<size_t>(y)*sourceStride;
        auto* b=output+static_cast<size_t>(y)*outputStride;
        for(uint32_t x=0;x<width;++x){
            const double yb=(b[x*4]*c[0]+b[x*4+1]*c[1]+b[x*4+2]*c[2])/255.0;
            const double restoredY=(yb-outputMean)*lumaGain+sourceMean;
            for(int channel=0;channel<3;++channel){
                const double original=b[x*4+channel]/255.0;
                const double chroma=original-yb;
                const double restored=restoredY+(chroma-outputChroma[channel])*
                    chromaGain+sourceChroma[channel];
                b[x*4+channel]=static_cast<uint8_t>(std::clamp(
                    (original+(restored-original)*amount)*255.0+0.5,0.0,255.0));
            }
            b[x*4+3]=a[x*4+3];
        }
    }
}

extern "C" __declspec(dllexport) void* dlssnr_create(
    const wchar_t* runtimePath, const wchar_t* corePath, uint32_t width,
    uint32_t height, int gpuIndex, int preset, char* error, int errorCapacity) {
    auto* session = new Session();
    std::string message;
    if (!session->Initialize(
            runtimePath, corePath, width, height, gpuIndex, preset, message)) {
        WriteError(error, errorCapacity, message); delete session; return nullptr;
    }
    WriteError(error, errorCapacity, "");
    return session;
}

extern "C" __declspec(dllexport) int dlssnr_process(
    void* handle, const uint8_t* input, uint32_t inputStride,
    const float* depth, uint32_t depthStride,
    const uint16_t* motion, uint32_t motionStride,
    const DlssnrSettingsC* settings,
    uint8_t* output, uint32_t outputStride,
    char* error, int errorCapacity) {
    if (!handle || !settings) { WriteError(error, errorCapacity, "Invalid arguments"); return 0; }
    std::string message;
    const bool ok = static_cast<Session*>(handle)->Process(input, inputStride, depth,
        depthStride, motion, motionStride, *settings, output, outputStride, message);
    WriteError(error, errorCapacity, message);
    return ok ? 1 : 0;
}

extern "C" __declspec(dllexport) void dlssnr_destroy(void* handle) {
    delete static_cast<Session*>(handle);
}
