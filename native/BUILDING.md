# Native bridge build

The shipped bridge was built for Windows x64 with MinGW-w64. Obtain `nvsdk_ngx.h`, `nvsdk_ngx_defs.h`, and `nvsdk_ngx_params.h` from the NVIDIA NGX SDK and place them in an include directory.

```powershell
g++ -std=c++20 -O2 -s -shared -static-libgcc -static-libstdc++ `
  -I C:\path\to\ngx-headers dlssnr_bridge.cpp `
  -o dlssnr_bridge.dll -ld3d11 -ld3d12 -ldxgi -ldxguid
```

The proprietary `nvngx_dlssnr.dll` is intentionally not linked or redistributed. It is selected by the user at runtime.
