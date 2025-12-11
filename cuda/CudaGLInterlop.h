#ifndef CUDA_GL_INTERLOP_H
#define CUDA_GL_INTERLOP_H

#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// GL 1.2+ constants not defined in legacy gl.h
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif

#include "vec3.h"

class CudaGLInterlop {
public:
    CudaGLInterlop();
    ~CudaGLInterlop();

    // Initialize with dimensions (creates GL texture and PBO)
    bool initialize(int width, int height);

    // Resize texture and PBO
    void resize(int width, int height);

    // Copy CUDA framebuffer to GL texture (GPU-only, zero-copy)
    void copyFramebufferToTexture(vec3* d_framebuffer, int width, int height);

    // Copy float4 framebuffer (for OptiX compatibility)
    void copyFramebufferToTexture(float4* d_framebuffer, int width, int height);

    // Batch rendering: Convert GPU framebuffer to CPU RGBA bytes (for saving to file)
    // Does gamma correction and float→byte conversion on GPU, then copies smaller byte buffer to CPU
    // Returns pointer to internal buffer (valid until next call or cleanup)
    unsigned char* convertToCPUImage(vec3* d_framebuffer, int width, int height, bool flipY = true);
    unsigned char* convertToCPUImage(float4* d_framebuffer, int width, int height, bool flipY = true);

    // Get texture ID
    GLuint getTexture() const { return glTexture; }

    // Clean up
    void cleanup();

private:
    GLuint glTexture;
    GLuint glPBO;                         // Pixel Buffer Object for CUDA-GL interop
    cudaGraphicsResource* cudaPBOResource; // CUDA resource for PBO
    int texWidth, texHeight;
    bool initialized;
    bool usePBO;                          // Whether PBO interop is available

    // Fallback for when PBO doesn't work
    unsigned char* h_pixels;

    // Batch rendering buffers
    unsigned char* d_batchBuffer;  // GPU byte buffer for batch conversion
    unsigned char* h_batchBuffer;  // CPU byte buffer for batch output
    int batchBufferSize;
};

#endif
