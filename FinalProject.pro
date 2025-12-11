QT += core gui opengl openglwidgets



CONFIG += c++17
CONFIG -= app_bundle

TEMPLATE = app

QMAKE_MSC_VER = 1939

# ============= CUDA Configuration =============
CUDA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

# Include CUDA headers
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += $$PWD

# Link CUDA libraries, OpenGL, and Windows libs for OptiX
LIBS += -L$$CUDA_DIR/lib/x64 -lcudart -lcurand -lopengl32 -ladvapi32

# RTX 5080 (Blackwell) = sm_120
CUDA_ARCH = sm_120

# ============= OptiX Configuration =============
OPTIX_DIR = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"

# Include OptiX headers
INCLUDEPATH += $$OPTIX_DIR/include

# ============= Source Files =============
SOURCES += \
    main.cpp \
    qt/Viewport.cpp \
    graphics/Mesh.cpp \
    graphics/Shader.cpp \
    graphics/Texture.cpp

# ============= Header Files =============
HEADERS += \
    qt/Viewport.h \
    qt/RenderSettingsPanelBase.h \
    qt/RenderSettingsPanel.h \
    math/Mathematics.h \
    math/Matrix3.h \
    math/Matrix4.h \
    math/Quaternion.h \
    math/Transformation.h \
    math/Vector2.h \
    math/Vector3.h \
    math/Vector4.h \
    graphics/Camera.h \
    graphics/Color3.h \
    graphics/Color4.h \
    graphics/Face.h \
    graphics/Light.h \
    graphics/Mesh.h \
    graphics/Shader.h \
    graphics/Texture.h \
    graphics/MouseCamera.h \
    graphics/Vertex.h \
    raytracer/rtweekend.h \
    raytracer/vec3.h \
    raytracer/ray.h \
    raytracer/color.h \
    raytracer/interval.h \
    raytracer/hittable.h \
    raytracer/hittable_list.h \
    raytracer/sphere.h \
    raytracer/material.h \
    raytracer/camera.h \
    raytracer/triangle.h \
    raytracer/aabb.h \
    raytracer/bvh.h \
    raytracer/scene_builder.h \
    raytracer/renderer.h \
    cuda/vec3.h \
    cuda/ray.h \
    cuda/aabb.h \
    cuda/sphere.h \
    cuda/hitable.h \
    cuda/hitable_list.h \
    cuda/camera.h \
    cuda/material.h \
    cuda/bvh.h \
    cuda/CudaRenderer.h \
    cuda/CudaGLInterlop.h \
    cuda/OptixRenderer.h \
    usd/USDAParser.h

# ============= CUDA Sources =============
CUDA_SOURCES = \
    cuda/CudaRenderer.cu \
    cuda/CudaGLInterlop.cu \
    cuda/OptixRenderer.cu

# ============= NVCC Compiler Rule =============
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.obj

# Use /MDd for debug, /MD for release
# Both use optimization flags for performance
CONFIG(debug, debug|release) {
    CUDA_RUNTIME = /MDd
    CUDA_OPT = -O3 -use_fast_math --expt-relaxed-constexpr
} else {
    CUDA_RUNTIME = /MD
    CUDA_OPT = -O3 -use_fast_math --expt-relaxed-constexpr
}

cuda.commands = \"$$CUDA_DIR/bin/nvcc.exe\" \
    -c ${QMAKE_FILE_NAME} \
    -o ${QMAKE_FILE_OUT} \
    -I\"$$CUDA_DIR/include\" \
    -I\"$$OPTIX_DIR/include\" \
    -I\"$$PWD\" \
    -std=c++14 \
    --gpu-architecture=$$CUDA_ARCH \
    $$CUDA_OPT \
    -Xcompiler \"/EHsc /W3 /nologo /FS $$CUDA_RUNTIME\"
cuda.dependency_type = TYPE_C
cuda.CONFIG += no_link
QMAKE_EXTRA_COMPILERS += cuda

# Link the CUDA object files
PRE_TARGETDEPS += CudaRenderer_cuda.obj CudaGLInterlop_cuda.obj OptixRenderer_cuda.obj
LIBS += CudaRenderer_cuda.obj CudaGLInterlop_cuda.obj OptixRenderer_cuda.obj

# ============= OptiX PTX Compilation Rule =============
# OptiX programs compile to PTX (loaded at runtime)
OPTIX_SOURCES = \
    cuda/optix_programs.cu

ptx.input = OPTIX_SOURCES
ptx.output = ${QMAKE_FILE_BASE}.ptx
ptx.commands = \"$$CUDA_DIR/bin/nvcc.exe\" \
    -ptx ${QMAKE_FILE_NAME} \
    -o ${QMAKE_FILE_OUT} \
    -I\"$$CUDA_DIR/include\" \
    -I\"$$OPTIX_DIR/include\" \
    -I\"$$PWD\" \
    -std=c++14 \
    --gpu-architecture=$$CUDA_ARCH \
    -use_fast_math \
    -Xcompiler \"/EHsc /W3 /nologo /FS $$CUDA_RUNTIME\"
ptx.dependency_type = TYPE_C
ptx.CONFIG += no_link target_predeps
QMAKE_EXTRA_COMPILERS += ptx

# ============= Post-build: Copy assets =============
# Disabled for now - copy manually if needed
# win32 {
#     CONFIG(debug, debug|release) {
#         QMAKE_POST_LINK += $$QMAKE_COPY_DIR $$shell_quote($$PWD/assets) $$shell_quote($$OUT_PWD/debug)
#     } else {
#         QMAKE_POST_LINK += $$QMAKE_COPY_DIR $$shell_quote($$PWD/assets) $$shell_quote($$OUT_PWD/release)
#     }
# }
