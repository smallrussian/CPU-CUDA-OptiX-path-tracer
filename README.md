# Real-Time Path Tracer with CUDA and OptiX

A Qt-based ray tracing application featuring four rendering backends: OpenGL rasterization, CPU path tracing, CUDA software ray tracing, and OptiX hardware-accelerated ray tracing.

## Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with RT cores (RTX 20-series or newer recommended)
- **NVIDIA Driver** 535.0 or newer

## Dependencies

### Qt 6.10.0 (MSVC 2022)

1. In QT Creator go to Tools -> QT Maintenance Tool -> Start Maintenence Tool -> SElect Add Or Remove Components -> Then under QT select 6.10.1 -> MSVC 2022 (64 Bit) -> Click Next and on the next page click update
2. Make sure to have the project use Desktop Qt 6.10.1 MSVC 2022 64bit kit instead of any other QT version, this being on first time setup

### CUDA Toolkit 12.x

1. Download from <https://developer.nvidia.com/cuda-downloads>
2. Install with default options
3. I used CUDA 12.8 for development, but other 12.x versions should work as well, though I cannot verify this.

### OptiX 9.0 SDK

1. Download and install from <https://developer.nvidia.com/designworks/optix/download>

### FFmpeg (for video export)

1. Run Command Prompt or Powershell as Administrator (it will probably work without elevated privileges, but just to be sure)
2. Run `winget install ffmpeg`

## Building

1. Open `FinalProject.pro` and `OpenGL.pro` in QT Creator
2. Configure the project to use the Desktop Qt 6.10.1 MSVC 2022 64bit kit
3. Make sure to select the Debug or Release configuration and NOT the Release (imported) configuration
4. Build both projects (Build -> Build All Projects)
5. BOTH PROJECTS MUST BE BUILT IN ORDER FOR THE APPLICATION TO WORK PROPERLY

## Usage

1. Run `PhongShading.exe` for CUDA/OptiX rendering
2. Run `OpenGLRayTracer.exe` for OpenGL preview with CPU ray tracing
3. Load scenes via File → Open (USDA format) (scenes are located in `assets/scenes/`) or in `scenes` folder in the program directory.
4. Switch render modes via the dropdown in the settings panel
5. Certain aspects of the raytracer can be configured in [`config.ini`](config.ini) program directory
6. Use mouse to orbit camera (right click-drag), zoom (scroll)
7. **Note:** *Some of the scenes in CUDA may start at the incorrect camera position, readjusting may be necessary.*

## Scene Format

1. The application uses USDA (Universal Scene Description ASCII) files. Example scenes are in the `assetss/scenes/` folder in the working directory when running from QT Creator.
2. The default working directory when running from QT Creator is `final project\build\Desktop_Qt_6_10_1_MSVC2022_64bit-Debug`.
