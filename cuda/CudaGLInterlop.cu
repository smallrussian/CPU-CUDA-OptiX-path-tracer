#include "CudaGLInterlop.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cstring>

// OpenGL extension function pointers (need to load at runtime on Windows)
typedef void (APIENTRY *PFNGLGENBUFFERSPROC)(GLsizei n, GLuint *buffers);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC)(GLsizei n, const GLuint *buffers);
typedef void (APIENTRY *PFNGLBINDBUFFERPROC)(GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC)(GLenum target, ptrdiff_t size, const void *data, GLenum usage);

static PFNGLGENBUFFERSPROC glGenBuffersPtr = nullptr;
static PFNGLDELETEBUFFERSPROC glDeleteBuffersPtr = nullptr;
static PFNGLBINDBUFFERPROC glBindBufferPtr = nullptr;
static PFNGLBUFFERDATAPROC glBufferDataPtr = nullptr;

static bool loadGLExtensions() {
    if (glGenBuffersPtr) return true;  // Already loaded

    glGenBuffersPtr = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    glDeleteBuffersPtr = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
    glBindBufferPtr = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    glBufferDataPtr = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");

    return (glGenBuffersPtr && glDeleteBuffersPtr && glBindBufferPtr && glBufferDataPtr);
}

#define checkCudaErrors(val) check_cuda_interop((val), #val, __FILE__, __LINE__)
static void check_cuda_interop(cudaError_t result, char const *const func,
                                const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' - " << cudaGetErrorString(result) << "\n";
    }
}

// CUDA kernel to convert float RGB to byte RGBA (runs entirely on GPU)
__global__ void convertFloatToRGBA(vec3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    vec3 color = input[idx];

    // Clamp and convert to 0-255
    unsigned char r = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.r())));
    unsigned char g = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.g())));
    unsigned char b = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.b())));

    int outIdx = idx * 4;
    output[outIdx + 0] = r;
    output[outIdx + 1] = g;
    output[outIdx + 2] = b;
    output[outIdx + 3] = 255;  // Alpha
}

// CUDA kernel to convert float4 RGBA to byte RGBA (for OptiX)
__global__ void convertFloat4ToRGBA(float4* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 color = input[idx];

    // Clamp and convert to 0-255
    unsigned char r = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.x)));
    unsigned char g = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.y)));
    unsigned char b = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.z)));

    int outIdx = idx * 4;
    output[outIdx + 0] = r;
    output[outIdx + 1] = g;
    output[outIdx + 2] = b;
    output[outIdx + 3] = 255;  // Alpha
}

// CUDA kernel for batch rendering: convert vec3 to RGBA with Y-flip
// NOTE: No gamma correction here - renderers (CUDA/OptiX) already apply gamma in their output
__global__ void convertFloatToRGBAFlipped(vec3* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Flip Y: GPU framebuffer has origin at bottom-left, image at top-left
    int srcIdx = (height - 1 - y) * width + x;
    vec3 color = input[srcIdx];

    // Clamp and convert to 0-255 (no gamma - already applied by renderer)
    unsigned char r = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.r())));
    unsigned char g = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.g())));
    unsigned char b = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.b())));

    int outIdx = (y * width + x) * 4;
    output[outIdx + 0] = r;
    output[outIdx + 1] = g;
    output[outIdx + 2] = b;
    output[outIdx + 3] = 255;
}

// CUDA kernel for batch rendering: convert float4 to RGBA with Y-flip
// NOTE: No gamma correction here - renderers (CUDA/OptiX) already apply gamma in their output
__global__ void convertFloat4ToRGBAFlipped(float4* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Flip Y: GPU framebuffer has origin at bottom-left, image at top-left
    int srcIdx = (height - 1 - y) * width + x;
    float4 color = input[srcIdx];

    // Clamp and convert to 0-255 (no gamma - already applied by renderer)
    unsigned char r = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.x)));
    unsigned char g = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.y)));
    unsigned char b = (unsigned char)(255.99f * fminf(1.0f, fmaxf(0.0f, color.z)));

    int outIdx = (y * width + x) * 4;
    output[outIdx + 0] = r;
    output[outIdx + 1] = g;
    output[outIdx + 2] = b;
    output[outIdx + 3] = 255;
}

CudaGLInterlop::CudaGLInterlop()
    : glTexture(0), glPBO(0), cudaPBOResource(nullptr),
      texWidth(0), texHeight(0), initialized(false), usePBO(false), h_pixels(nullptr),
      d_batchBuffer(nullptr), h_batchBuffer(nullptr), batchBufferSize(0) {}

CudaGLInterlop::~CudaGLInterlop() {
    cleanup();
}

bool CudaGLInterlop::initialize(int width, int height) {
    texWidth = width;
    texHeight = height;

    // Load GL extensions
    if (!loadGLExtensions()) {
        std::cerr << "[CudaGLInterlop] Failed to load GL extensions, using fallback" << std::endl;
        usePBO = false;
    } else {
        usePBO = true;
    }

    // Create GL texture
    glGenTextures(1, &glTexture);
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, texWidth, texHeight, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (usePBO) {
        // Create PBO for CUDA-GL interop
        glGenBuffersPtr(1, &glPBO);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, glPBO);
        glBufferDataPtr(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, 0);

        // Register PBO with CUDA
        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPBOResource, glPBO,
                                                        cudaGraphicsMapFlagsWriteDiscard);
        if (err != cudaSuccess) {
            std::cerr << "[CudaGLInterlop] Failed to register PBO: " << cudaGetErrorString(err) << std::endl;
            usePBO = false;
            glDeleteBuffersPtr(1, &glPBO);
            glPBO = 0;
        } else {
            std::cout << "[CudaGLInterlop] PBO interop initialized (zero-copy GPU path)" << std::endl;
        }
    }

    if (!usePBO) {
        // Fallback: allocate host buffer
        h_pixels = new unsigned char[texWidth * texHeight * 4];
        std::cout << "[CudaGLInterlop] Using fallback CPU path" << std::endl;
    }

    initialized = true;
    return true;
}

void CudaGLInterlop::resize(int width, int height) {
    if (width == texWidth && height == texHeight) return;

    // Cleanup old resources
    if (usePBO && cudaPBOResource) {
        cudaGraphicsUnregisterResource(cudaPBOResource);
        cudaPBOResource = nullptr;
    }
    if (glPBO) {
        glDeleteBuffersPtr(1, &glPBO);
        glPBO = 0;
    }
    if (h_pixels) {
        delete[] h_pixels;
        h_pixels = nullptr;
    }

    texWidth = width;
    texHeight = height;

    // Resize texture
    glBindTexture(GL_TEXTURE_2D, glTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (usePBO && glGenBuffersPtr) {
        // Recreate PBO
        glGenBuffersPtr(1, &glPBO);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, glPBO);
        glBufferDataPtr(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_STREAM_DRAW);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, 0);

        cudaError_t err = cudaGraphicsGLRegisterBuffer(&cudaPBOResource, glPBO,
                                                        cudaGraphicsMapFlagsWriteDiscard);
        if (err != cudaSuccess) {
            std::cerr << "[CudaGLInterlop] Failed to re-register PBO: " << cudaGetErrorString(err) << std::endl;
            usePBO = false;
        }
    }

    if (!usePBO) {
        h_pixels = new unsigned char[width * height * 4];
    }
}

void CudaGLInterlop::copyFramebufferToTexture(vec3* d_framebuffer, int width, int height) {
    if (!initialized || !d_framebuffer) return;

    if (usePBO && cudaPBOResource) {
        // GPU-only path: CUDA writes directly to PBO

        // Map PBO for CUDA
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBOResource, 0));

        unsigned char* d_pbo;
        size_t pboSize;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &pboSize, cudaPBOResource));

        // Run kernel to convert float RGB to byte RGBA
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertFloatToRGBA<<<gridSize, blockSize>>>(d_framebuffer, d_pbo, width, height);

        // Unmap PBO
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));

        // Copy PBO to texture
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, glPBO);
        glBindTexture(GL_TEXTURE_2D, glTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, 0);  // 0 = offset in PBO
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, 0);
    } else {
        // Fallback: CPU path
        int num_pixels = width * height;
        vec3* h_fb = new vec3[num_pixels];
        checkCudaErrors(cudaMemcpy(h_fb, d_framebuffer, num_pixels * sizeof(vec3),
                                   cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_pixels; i++) {
            int r = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].r())));
            int g = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].g())));
            int b = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].b())));
            h_pixels[i * 4 + 0] = (unsigned char)r;
            h_pixels[i * 4 + 1] = (unsigned char)g;
            h_pixels[i * 4 + 2] = (unsigned char)b;
            h_pixels[i * 4 + 3] = 255;
        }
        delete[] h_fb;

        glBindTexture(GL_TEXTURE_2D, glTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, h_pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void CudaGLInterlop::copyFramebufferToTexture(float4* d_framebuffer, int width, int height) {
    if (!initialized || !d_framebuffer) return;

    if (usePBO && cudaPBOResource) {
        // GPU-only path: CUDA writes directly to PBO
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaPBOResource, 0));

        unsigned char* d_pbo;
        size_t pboSize;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_pbo, &pboSize, cudaPBOResource));

        // Run kernel to convert float4 RGBA to byte RGBA
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        convertFloat4ToRGBA<<<gridSize, blockSize>>>(d_framebuffer, d_pbo, width, height);

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaPBOResource, 0));

        // Copy PBO to texture
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, glPBO);
        glBindTexture(GL_TEXTURE_2D, glTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBufferPtr(GL_PIXEL_UNPACK_BUFFER, 0);
    } else {
        // Fallback: CPU path
        int num_pixels = width * height;
        float4* h_fb = new float4[num_pixels];
        checkCudaErrors(cudaMemcpy(h_fb, d_framebuffer, num_pixels * sizeof(float4),
                                   cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_pixels; i++) {
            int r = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].x)));
            int g = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].y)));
            int b = int(255.99f * fminf(1.0f, fmaxf(0.0f, h_fb[i].z)));
            h_pixels[i * 4 + 0] = (unsigned char)r;
            h_pixels[i * 4 + 1] = (unsigned char)g;
            h_pixels[i * 4 + 2] = (unsigned char)b;
            h_pixels[i * 4 + 3] = 255;
        }
        delete[] h_fb;

        glBindTexture(GL_TEXTURE_2D, glTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                        GL_RGBA, GL_UNSIGNED_BYTE, h_pixels);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void CudaGLInterlop::cleanup() {
    if (!initialized) return;

    if (cudaPBOResource) {
        cudaGraphicsUnregisterResource(cudaPBOResource);
        cudaPBOResource = nullptr;
    }

    if (glPBO && glDeleteBuffersPtr) {
        glDeleteBuffersPtr(1, &glPBO);
        glPBO = 0;
    }

    if (h_pixels) {
        delete[] h_pixels;
        h_pixels = nullptr;
    }

    if (d_batchBuffer) {
        cudaFree(d_batchBuffer);
        d_batchBuffer = nullptr;
    }

    if (h_batchBuffer) {
        delete[] h_batchBuffer;
        h_batchBuffer = nullptr;
    }
    batchBufferSize = 0;

    if (glTexture) {
        glDeleteTextures(1, &glTexture);
        glTexture = 0;
    }

    initialized = false;
}

unsigned char* CudaGLInterlop::convertToCPUImage(vec3* d_framebuffer, int width, int height, bool flipY) {
    if (!d_framebuffer) return nullptr;

    int requiredSize = width * height * 4;

    // Allocate/reallocate buffers if needed
    if (requiredSize > batchBufferSize) {
        if (d_batchBuffer) cudaFree(d_batchBuffer);
        if (h_batchBuffer) delete[] h_batchBuffer;

        checkCudaErrors(cudaMalloc(&d_batchBuffer, requiredSize));
        h_batchBuffer = new unsigned char[requiredSize];
        batchBufferSize = requiredSize;
    }

    // Run GPU kernel to convert float→byte with gamma correction
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    if (flipY) {
        convertFloatToRGBAFlipped<<<gridSize, blockSize>>>(d_framebuffer, d_batchBuffer, width, height);
    } else {
        convertFloatToRGBA<<<gridSize, blockSize>>>(d_framebuffer, d_batchBuffer, width, height);
    }

    // Copy byte buffer to CPU (4x smaller than float buffer!)
    checkCudaErrors(cudaMemcpy(h_batchBuffer, d_batchBuffer, requiredSize, cudaMemcpyDeviceToHost));

    return h_batchBuffer;
}

unsigned char* CudaGLInterlop::convertToCPUImage(float4* d_framebuffer, int width, int height, bool flipY) {
    if (!d_framebuffer) return nullptr;

    int requiredSize = width * height * 4;

    // Allocate/reallocate buffers if needed
    if (requiredSize > batchBufferSize) {
        if (d_batchBuffer) cudaFree(d_batchBuffer);
        if (h_batchBuffer) delete[] h_batchBuffer;

        checkCudaErrors(cudaMalloc(&d_batchBuffer, requiredSize));
        h_batchBuffer = new unsigned char[requiredSize];
        batchBufferSize = requiredSize;
    }

    // Run GPU kernel to convert float→byte with gamma correction
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    if (flipY) {
        convertFloat4ToRGBAFlipped<<<gridSize, blockSize>>>(d_framebuffer, d_batchBuffer, width, height);
    } else {
        convertFloat4ToRGBA<<<gridSize, blockSize>>>(d_framebuffer, d_batchBuffer, width, height);
    }

    // Copy byte buffer to CPU (4x smaller than float buffer!)
    checkCudaErrors(cudaMemcpy(h_batchBuffer, d_batchBuffer, requiredSize, cudaMemcpyDeviceToHost));

    return h_batchBuffer;
}
