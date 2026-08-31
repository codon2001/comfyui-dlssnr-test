# Native bridge build

The shipped bridge was built for Windows x64 with MinGW-w64. Obtain `nvsdk_ngx.h`, `nvsdk_ngx_defs.h`, and `nvsdk_ngx_params.h` from the NVIDIA NGX SDK and place them in an include directory.

```powershell
g++ -std=c++20 -O2 -s -shared -static-libgcc -static-libstdc++ `
  -I C:\path\to\ngx-headers dlssnr_bridge.cpp `
  -o dlssnr_bridge.dll -ld3d11 -ld3d12 -ldxgi -ldxguid
```

V1.4 also ships a D3D11/Media Foundation zero-copy video worker and a CPU
RGBA fallback worker:

```powershell
g++ -std=c++20 -O3 -s -static -municode -I C:\path\to\ngx-headers `
  video_worker_gpu.cpp -o dlssnr_video_worker.exe `
  -ld3d11 -ld3d12 -ldxgi -ldxguid -ld3dcompiler `
  -lmfplat -lmfreadwrite -lmfuuid -lmf -lole32 -luuid

g++ -std=c++20 -O3 -fopenmp -s -static -municode `
  -I C:\path\to\ngx-headers video_worker.cpp `
  -o dlssnr_video_worker_cpu.exe -ld3d11 -ld3d12 -ldxgi -ldxguid
```

The proprietary `nvngx_dlssnr.dll` is intentionally not linked or redistributed. It is selected by the user at runtime.
