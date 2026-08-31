#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "dlssnr_bridge.cpp"

namespace {

std::wstring Quote(const std::wstring& value) {
    return L"\"" + value + L"\"";
}

struct ChildProcess {
    PROCESS_INFORMATION process{};
    HANDLE parentPipe = nullptr;
};

bool SpawnWithPipe(const std::wstring& command, bool childWrites,
                   HANDLE job, ChildProcess& result, std::string& error) {
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE readPipe = nullptr;
    HANDLE writePipe = nullptr;
    if (!CreatePipe(&readPipe, &writePipe, &security, 4 * 1024 * 1024)) {
        error = "CreatePipe failed";
        return false;
    }
    result.parentPipe = childWrites ? readPipe : writePipe;
    HANDLE childPipe = childWrites ? writePipe : readPipe;
    SetHandleInformation(result.parentPipe, HANDLE_FLAG_INHERIT, 0);
    HANDLE nullHandle = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = childWrites ? nullHandle : childPipe;
    startup.hStdOutput = childWrites ? childPipe : nullHandle;
    startup.hStdError = nullHandle;
    std::vector<wchar_t> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back(0);
    const BOOL created = CreateProcessW(nullptr, mutableCommand.data(), nullptr,
        nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &result.process);
    CloseHandle(childPipe);
    if (nullHandle != INVALID_HANDLE_VALUE) CloseHandle(nullHandle);
    if (!created) {
        CloseHandle(result.parentPipe);
        result.parentPipe = nullptr;
        error = "CreateProcessW failed: " + std::to_string(GetLastError());
        return false;
    }
    if (job) AssignProcessToJobObject(job, result.process.hProcess);
    CloseHandle(result.process.hThread);
    result.process.hThread = nullptr;
    return true;
}

bool ReadFrame(HANDLE pipe, uint8_t* output, size_t size, bool& eof) {
    eof = false;
    size_t offset = 0;
    while (offset < size) {
        DWORD read = 0;
        const DWORD request = static_cast<DWORD>(
            std::min<size_t>(size - offset, 16 * 1024 * 1024));
        if (!ReadFile(pipe, output + offset, request, &read, nullptr) || !read) {
            eof = offset == 0;
            return eof;
        }
        offset += read;
    }
    return true;
}

bool WriteFrame(HANDLE pipe, const uint8_t* input, size_t size) {
    size_t offset = 0;
    while (offset < size) {
        DWORD written = 0;
        const DWORD request = static_cast<DWORD>(
            std::min<size_t>(size - offset, 16 * 1024 * 1024));
        if (!WriteFile(pipe, input + offset, request, &written, nullptr) || !written)
            return false;
        offset += written;
    }
    return true;
}

void RestoreSourceColor(const uint8_t* source, uint8_t* output,
                        uint32_t width, uint32_t height) {
    const size_t pixels = static_cast<size_t>(width) * height;
    double sourceMean = 0.0, outputMean = 0.0;
    double sourceSq = 0.0, outputSq = 0.0;
    double sourceChroma[3]{}, outputChroma[3]{};
    double sourceEnergy = 0.0, outputEnergy = 0.0;
    constexpr double c[3]{0.2126, 0.7152, 0.0722};
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sourceMean,outputMean,sourceSq,outputSq,sourceChroma[:3],outputChroma[:3])
    #endif
    for (int64_t index = 0; index < static_cast<int64_t>(pixels); ++index) {
        const auto* a = source + index * 4;
        const auto* b = output + index * 4;
        const double ya = (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]) / 255.0;
        const double yb = (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) / 255.0;
        sourceMean += ya; outputMean += yb;
        sourceSq += ya * ya; outputSq += yb * yb;
        for (int channel = 0; channel < 3; ++channel) {
            sourceChroma[channel] += a[channel] / 255.0 - ya;
            outputChroma[channel] += b[channel] / 255.0 - yb;
        }
    }
    sourceMean /= pixels; outputMean /= pixels;
    for (int channel = 0; channel < 3; ++channel) {
        sourceChroma[channel] /= pixels;
        outputChroma[channel] /= pixels;
    }
    const double sourceStd = std::sqrt(std::max(sourceSq / pixels - sourceMean * sourceMean, 1e-8));
    const double outputStd = std::sqrt(std::max(outputSq / pixels - outputMean * outputMean, 1e-8));
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:sourceEnergy,outputEnergy)
    #endif
    for (int64_t index = 0; index < static_cast<int64_t>(pixels); ++index) {
        const auto* a = source + index * 4;
        const auto* b = output + index * 4;
        const double ya = (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]) / 255.0;
        const double yb = (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) / 255.0;
        for (int channel = 0; channel < 3; ++channel) {
            const double ca = a[channel] / 255.0 - ya - sourceChroma[channel];
            const double cb = b[channel] / 255.0 - yb - outputChroma[channel];
            sourceEnergy += ca * ca;
            outputEnergy += cb * cb;
        }
    }
    const double lumaGain = std::clamp(sourceStd / outputStd, 0.5, 2.0);
    const double chromaGain = std::clamp(std::sqrt(std::max(sourceEnergy, 1e-8) /
        std::max(outputEnergy, 1e-8)), 0.5, 2.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t index = 0; index < static_cast<int64_t>(pixels); ++index) {
        const auto* a = source + index * 4;
        auto* b = output + index * 4;
        const double yb = (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) / 255.0;
        const double restoredY = (yb - outputMean) * lumaGain + sourceMean;
        for (int channel = 0; channel < 3; ++channel) {
            const double chroma = b[channel] / 255.0 - yb;
            const double restored = restoredY +
                (chroma - outputChroma[channel]) * chromaGain + sourceChroma[channel];
            b[channel] = static_cast<uint8_t>(std::clamp(restored * 255.0 + 0.5, 0.0, 255.0));
        }
        b[3] = a[3];
    }
}

void WritePreview(const std::filesystem::path& path, const uint8_t* source,
                  const uint8_t* processed, uint32_t width, uint32_t height) {
    if (path.empty()) return;
    const uint32_t previewWidth = std::min<uint32_t>(width, 480);
    const uint32_t previewHeight = std::max<uint32_t>(1,
        static_cast<uint32_t>(static_cast<uint64_t>(height) * previewWidth / width));
    const std::filesystem::path temporary = path.wstring() + L".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) return;
    stream << "P6\n" << previewWidth * 2 + 4 << " " << previewHeight << "\n255\n";
    for (uint32_t y = 0; y < previewHeight; ++y) {
        const uint32_t sourceY = static_cast<uint32_t>(
            static_cast<uint64_t>(y) * height / previewHeight);
        for (int side = 0; side < 2; ++side) {
            const uint8_t* rgba = side == 0 ? source : processed;
            for (uint32_t x = 0; x < previewWidth; ++x) {
                const uint32_t sourceX = static_cast<uint32_t>(
                    static_cast<uint64_t>(x) * width / previewWidth);
                const auto* pixel = rgba +
                    (static_cast<size_t>(sourceY) * width + sourceX) * 4;
                stream.write(reinterpret_cast<const char*>(pixel), 3);
            }
            if (side == 0) {
                const uint8_t divider[12]{255, 255, 255, 32, 32, 32,
                                          32, 32, 32, 255, 255, 255};
                stream.write(reinterpret_cast<const char*>(divider), sizeof(divider));
            }
        }
    }
    stream.close();
    MoveFileExW(temporary.c_str(), path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
}

double ParseDouble(const wchar_t* value) { return std::wcstod(value, nullptr); }
int ParseInt(const wchar_t* value) { return static_cast<int>(std::wcstol(value, nullptr, 10)); }

} // namespace

int wmain(int argc, wchar_t** argv) {
    if (argc != 22) {
        std::fprintf(stderr, "video_worker: expected 21 arguments\n");
        return 2;
    }
    const std::wstring ffmpeg = argv[1];
    const std::wstring input = argv[2];
    const std::wstring output = argv[3];
    const std::wstring runtime = argv[4];
    const std::wstring core = argv[5];
    const uint32_t width = static_cast<uint32_t>(ParseInt(argv[6]));
    const uint32_t height = static_cast<uint32_t>(ParseInt(argv[7]));
    const std::wstring fps = argv[8];
    const int gpu = ParseInt(argv[9]);
    const int preset = ParseInt(argv[10]);
    DlssnrSettingsC settings{};
    settings.style = ParseInt(argv[11]);
    settings.intensity = static_cast<float>(ParseDouble(argv[12]));
    settings.localTone = static_cast<float>(ParseDouble(argv[13]));
    settings.localStructure = static_cast<float>(ParseDouble(argv[14]));
    settings.skinStructure = static_cast<float>(ParseDouble(argv[15]));
    settings.autoMask = ParseInt(argv[16]);
    settings.uiCorrection = ParseInt(argv[17]);
    settings.paperWhiteScale = static_cast<float>(ParseDouble(argv[18]));
    settings.depthInverted = 1;
    settings.motionScaleX = 1.0f;
    settings.motionScaleY = 1.0f;
    settings.colorTransfer = 0;
    const std::filesystem::path preview = argv[19];
    const std::filesystem::path stopSignal = argv[20];
    const double speedMultiplier = std::max(0.0, ParseDouble(argv[21]));
    const double sourceFps = std::max(0.01, ParseDouble(argv[8]));

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job) SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                      &limits, sizeof(limits));
    const std::wstring decoderCommand = Quote(ffmpeg) +
        L" -hide_banner -loglevel error -hwaccel cuda -i " + Quote(input) +
        L" -map 0:v:0 -an -sn -dn -f rawvideo -pix_fmt rgba pipe:1";
    const std::wstring encoderCommand = Quote(ffmpeg) +
        L" -hide_banner -loglevel error -f rawvideo -pix_fmt rgba -s " +
        std::to_wstring(width) + L"x" + std::to_wstring(height) +
        L" -r " + fps + L" -i pipe:0 -an -c:v h264_nvenc -preset p4 "
        L"-tune hq -rc vbr -cq 18 -b:v 0 -pix_fmt yuv420p -y " + Quote(output);
    ChildProcess decoder{}, encoder{};
    std::string error;
    if (!SpawnWithPipe(decoderCommand, true, job, decoder, error) ||
        !SpawnWithPipe(encoderCommand, false, job, encoder, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        if (job) CloseHandle(job);
        return 3;
    }
    Session session;
    if (!session.Initialize(runtime.c_str(), core.c_str(), width, height,
                            gpu, preset, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        CloseHandle(job);
        return 4;
    }
    const size_t frameBytes = static_cast<size_t>(width) * height * 4;
    std::vector<uint8_t> source(frameBytes), processed(frameBytes);
    uint64_t frames = 0;
    double decodeSeconds = 0.0, dlssSeconds = 0.0;
    double colorSeconds = 0.0, encodeSeconds = 0.0;
    const auto started = std::chrono::steady_clock::now();
    while (true) {
        if (!stopSignal.empty() && std::filesystem::exists(stopSignal)) break;
        bool eof = false;
        auto stageStarted = std::chrono::steady_clock::now();
        if (!ReadFrame(decoder.parentPipe, source.data(), frameBytes, eof)) {
            if (!eof) error = "decoder returned an incomplete frame";
            break;
        }
        if (eof) break;
        decodeSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stageStarted).count();
        settings.reset = frames == 0 ? 1 : 0;
        stageStarted = std::chrono::steady_clock::now();
        if (!session.Process(source.data(), width * 4, nullptr, 0, nullptr, 0,
                             settings, processed.data(), width * 4, error)) break;
        dlssSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stageStarted).count();
        stageStarted = std::chrono::steady_clock::now();
        RestoreSourceColor(source.data(), processed.data(), width, height);
        colorSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stageStarted).count();
        stageStarted = std::chrono::steady_clock::now();
        if (!WriteFrame(encoder.parentPipe, processed.data(), frameBytes)) {
            error = "encoder pipe closed";
            break;
        }
        encodeSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - stageStarted).count();
        ++frames;
        if (frames == 1 || frames % 15 == 0) {
            WritePreview(preview, source.data(), processed.data(), width, height);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            std::printf("PROGRESS %llu %.3f\n",
                        static_cast<unsigned long long>(frames), frames / std::max(seconds, 1e-6));
            std::fflush(stdout);
        }
        if (speedMultiplier > 0.0) {
            const auto target = started + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(frames /
                        (sourceFps * speedMultiplier)));
            std::this_thread::sleep_until(target);
        }
    }
    CloseHandle(decoder.parentPipe);
    CloseHandle(encoder.parentPipe);
    WaitForSingleObject(decoder.process.hProcess, INFINITE);
    WaitForSingleObject(encoder.process.hProcess, INFINITE);
    DWORD decoderExit = 1, encoderExit = 1;
    GetExitCodeProcess(decoder.process.hProcess, &decoderExit);
    GetExitCodeProcess(encoder.process.hProcess, &encoderExit);
    CloseHandle(decoder.process.hProcess);
    CloseHandle(encoder.process.hProcess);
    if (job) CloseHandle(job);
    if (!error.empty() || decoderExit != 0 || encoderExit != 0) {
        std::fprintf(stderr, "video pipeline failed: %s decoder=%lu encoder=%lu\n",
                     error.c_str(), decoderExit, encoderExit);
        return 5;
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::printf("DONE %llu %.3f\n", static_cast<unsigned long long>(frames),
                frames / std::max(seconds, 1e-6));
    const double divisor = std::max<double>(frames, 1.0);
    std::printf("STAGES %.3f %.3f %.3f %.3f\n",
                decodeSeconds * 1000.0 / divisor,
                dlssSeconds * 1000.0 / divisor,
                colorSeconds * 1000.0 / divisor,
                encodeSeconds * 1000.0 / divisor);
    return frames ? 0 : 6;
}
