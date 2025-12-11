#include "OptixRenderer.h"
#include "OptixConfig.h"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <fstream>
#include <sstream>
#include <iostream>

// Error checking macros
#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX Error: " << optixGetErrorName(res) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define OPTIX_CHECK_VOID(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX Error: " << optixGetErrorName(res) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
}

// CUDA kernel for frame accumulation
__global__ void accumulateKernel(float4* accumBuffer, float4* newFrame, int width, int height, unsigned int frameCount) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 newSample = newFrame[idx];

    if (frameCount == 0) {
        // First frame - just copy
        accumBuffer[idx] = newSample;
    } else {
        // Accumulate: running average
        float4 accum = accumBuffer[idx];
        float weight = 1.0f / (frameCount + 1);
        accumBuffer[idx] = make_float4(
            accum.x * (1.0f - weight) + newSample.x * weight,
            accum.y * (1.0f - weight) + newSample.y * weight,
            accum.z * (1.0f - weight) + newSample.z * weight,
            1.0f
        );
    }
}

// Copy accumulated buffer to output
__global__ void copyAccumToOutputKernel(float4* output, float4* accumBuffer, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    output[idx] = accumBuffer[idx];
}

OptixRenderer::OptixRenderer() {}

OptixRenderer::~OptixRenderer() {
    cleanup();
}

bool OptixRenderer::initialize(int w, int h) {
    width = w;
    height = h;

    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    if (!initContext()) return false;

    std::string ptxCode;
    if (!loadPTX(ptxCode)) return false;
    if (!createModule(ptxCode)) return false;
    if (!createProgramGroups()) return false;
    if (!createPipeline()) return false;
    if (!createSBT()) return false;

    allocateFramebuffer();

    // Allocate params on device
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(OptixParams)));

    // Ensure params are properly initialized
    memset(&params, 0, sizeof(OptixParams));  // Zero all fields first
    params.flowBuffer = nullptr;  // Motion vectors (Phase 1: not allocated yet)
    // Previous camera vectors initialized to zero (will be set on first frame)
    params.prev_cam_eye = make_float3(0.0f, 0.0f, 0.0f);
    params.prev_cam_u = make_float3(0.0f, 0.0f, 0.0f);
    params.prev_cam_v = make_float3(0.0f, 0.0f, 0.0f);
    params.prev_cam_w = make_float3(0.0f, 0.0f, 0.0f);
    params.lights = nullptr;
    params.num_lights = 0;
    params.frame_number = 0;
    params.samples_per_pixel = OPTIX_DEFAULT_SAMPLES_PER_PIXEL;
    params.useFlowBuffer = false;
    params.traversable = 0;

    // Denoiser disabled - skip initialization
    // if (!initDenoiser()) {
    //     std::cerr << "[OptixRenderer] Denoiser init failed - continuing without denoising" << std::endl;
    // }

    initialized = true;
    std::cout << "[OptixRenderer] Initialized successfully" << std::endl;
    return true;
}

bool OptixRenderer::initContext() {
    CUcontext cuCtx = 0; // Use current context

    // Query GPU properties to verify RT core support
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    std::cout << "========================================" << std::endl;
    std::cout << "[OptiX] GPU: " << props.name << std::endl;
    std::cout << "[OptiX] Compute Capability: " << props.major << "." << props.minor << std::endl;

    // RT Cores available on:
    // - Turing (SM 7.5): RTX 20xx series
    // - Ampere (SM 8.6): RTX 30xx series
    // - Ada Lovelace (SM 8.9): RTX 40xx series
    // - Blackwell (SM 10.0+): RTX 50xx series
    bool hasRTCores = (props.major >= 7 && props.minor >= 5) || props.major >= 8;

    if (hasRTCores) {
        std::cout << "[OptiX] RT Cores: AVAILABLE (Hardware Ray Tracing)" << std::endl;
    } else {
        std::cout << "[OptiX] RT Cores: NOT AVAILABLE (Software Fallback)" << std::endl;
    }

    std::cout << "[OptiX] Using OPTIX_PRIMITIVE_TYPE_SPHERE (Hardware BVH)" << std::endl;
    std::cout << "========================================" << std::endl;

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = contextLogCallback;
    options.logCallbackLevel = 4;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    return true;
}

bool OptixRenderer::loadPTX(std::string& ptxCode) {
    std::ifstream file("optix_programs.ptx");
    if (!file.is_open()) {
        std::cerr << "Failed to open optix_programs.ptx" << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    ptxCode = buffer.str();
    return true;
}

bool OptixRenderer::createModule(const std::string& ptxCode) {
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.usesMotionBlur = false;
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = OPTIX_NUM_PAYLOADS;
    pipelineOptions.numAttributeValues = OPTIX_NUM_ATTRIBUTES;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    // Support both sphere and triangle primitives
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    size_t logSize = sizeof(log);

    // Get builtin sphere intersection module
    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.usesMotionBlur = false;
    builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OPTIX_CHECK(optixBuiltinISModuleGet(context, &moduleOptions, &pipelineOptions, &builtinISOptions, &sphereModule));

    OPTIX_CHECK(optixModuleCreate(
        context,
        &moduleOptions,
        &pipelineOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &logSize,
        &module
    ));

    if (logSize > 1) std::cerr << "Module log: " << log << std::endl;
    return true;
}

bool OptixRenderer::createProgramGroups() {
    OptixProgramGroupOptions pgOptions = {};
    char log[2048];
    size_t logSize;

    // Raygen
    OptixProgramGroupDesc raygenDesc = {};
    raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenDesc.raygen.module = module;
    raygenDesc.raygen.entryFunctionName = "__raygen__rg";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygenDesc, 1, &pgOptions, log, &logSize, &raygenPG));

    // Miss - primary rays (sky)
    OptixProgramGroupDesc missDesc = {};
    missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missDesc.miss.module = module;
    missDesc.miss.entryFunctionName = "__miss__ms";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &missDesc, 1, &pgOptions, log, &logSize, &missPG));

    // Miss - shadow rays
    OptixProgramGroupDesc missShadowDesc = {};
    missShadowDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missShadowDesc.miss.module = module;
    missShadowDesc.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &missShadowDesc, 1, &pgOptions, log, &logSize, &missShadowPG));

    // Hit group for spheres - uses builtin sphere intersection
    OptixProgramGroupDesc hitSphereDesc = {};
    hitSphereDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitSphereDesc.hitgroup.moduleCH = module;
    hitSphereDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitSphereDesc.hitgroup.moduleIS = sphereModule;  // Builtin sphere intersection
    hitSphereDesc.hitgroup.entryFunctionNameIS = nullptr;  // Builtin, no custom function
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitSphereDesc, 1, &pgOptions, log, &logSize, &hitgroupSpherePG));

    // Hit group for triangles - uses builtin triangle intersection
    OptixProgramGroupDesc hitTriangleDesc = {};
    hitTriangleDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitTriangleDesc.hitgroup.moduleCH = module;
    hitTriangleDesc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
    hitTriangleDesc.hitgroup.moduleIS = nullptr;  // Builtin triangle intersection (no IS needed)
    hitTriangleDesc.hitgroup.entryFunctionNameIS = nullptr;
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitTriangleDesc, 1, &pgOptions, log, &logSize, &hitgroupTrianglePG));

    return true;
}

bool OptixRenderer::createPipeline() {
    OptixProgramGroup programGroups[] = { raygenPG, missPG, missShadowPG, hitgroupSpherePG, hitgroupTrianglePG };

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = OPTIX_MAX_TRACE_DEPTH;

    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = OPTIX_NUM_PAYLOADS;
    pipelineOptions.numAttributeValues = OPTIX_NUM_ATTRIBUTES;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char log[2048];
    size_t logSize = sizeof(log);

    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipelineOptions,
        &linkOptions,
        programGroups, 5,
        log, &logSize,
        &pipeline
    ));

    // Set stack sizes
    OptixStackSizes stackSizes = {};
    for (auto pg : programGroups) {
        OPTIX_CHECK(optixProgramGroupGetStackSize(pg, &stackSizes, pipeline));
    }

    uint32_t directCallableStackSizeFromTraversal = 0;
    uint32_t directCallableStackSizeFromState = 0;
    // Stack for recursion: raygen + miss + closesthit * maxTraceDepth
    uint32_t continuationStackSize = stackSizes.cssRG + stackSizes.cssMS +
        (stackSizes.cssCH * OPTIX_MAX_TRACE_DEPTH);

    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        2  // maxTraversableGraphDepth - 2 for IAS->GAS
    ));

    return true;
}

bool OptixRenderer::createSBT() {
    // Raygen record
    RayGenRecord raygenRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &raygenRecord));
    CUDA_CHECK(cudaMalloc((void**)&d_raygenRecord, sizeof(RayGenRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_raygenRecord, &raygenRecord, sizeof(RayGenRecord), cudaMemcpyHostToDevice));

    // Miss records - 2 entries: [0] = primary rays (sky), [1] = shadow rays
    MissRecord missRecords[2] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &missRecords[0]));
    missRecords[0].bg_color = make_float3(0.5f, 0.7f, 1.0f); // Sky blue (unused, computed in shader)
    OPTIX_CHECK(optixSbtRecordPackHeader(missShadowPG, &missRecords[1]));
    missRecords[1].bg_color = make_float3(0.0f, 0.0f, 0.0f); // Not used for shadow
    CUDA_CHECK(cudaMalloc((void**)&d_missRecord, 2 * sizeof(MissRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_missRecord, missRecords, 2 * sizeof(MissRecord), cudaMemcpyHostToDevice));

    // Hitgroup record (will be replaced when scene is built with materials)
    HitGroupRecord hitRecord = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG, &hitRecord));
    hitRecord.material.type = 0;  // Lambertian
    hitRecord.material.albedo = make_float3(0.8f, 0.3f, 0.3f);  // Red
    hitRecord.material.fuzz_or_ior = 0.0f;
    CUDA_CHECK(cudaMalloc((void**)&d_hitgroupRecord, sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_hitgroupRecord, &hitRecord, sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

    // Fill SBT
    sbt.raygenRecord = d_raygenRecord;
    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = 2;  // Two miss records: primary and shadow
    sbt.hitgroupRecordBase = d_hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = 1;

    return true;
}

void OptixRenderer::allocateFramebuffer() {
    freeFramebuffer();

    // Render at full resolution (denoiser disabled)
    cudaMalloc((void**)&d_framebuffer, width * height * sizeof(float4));
    cudaMemset(d_framebuffer, 0, width * height * sizeof(float4));

    // Accumulation buffer
    cudaMalloc((void**)&d_accumBuffer, width * height * sizeof(float4));
    cudaMemset(d_accumBuffer, 0, width * height * sizeof(float4));
    accumFrameCount = 0;

    // Motion vector buffer disabled - not needed without temporal denoiser
    d_flowBuffer = nullptr;
}

void OptixRenderer::freeFramebuffer() {
    if (d_framebuffer) {
        cudaFree(d_framebuffer);
        d_framebuffer = nullptr;
    }
    if (d_accumBuffer) {
        cudaFree(d_accumBuffer);
        d_accumBuffer = nullptr;
    }
    accumFrameCount = 0;
}

void OptixRenderer::resize(int w, int h) {
    width = w;
    height = h;
    allocateFramebuffer();

    // Denoiser disabled - skip reinitialization
    // if (denoiser) {
    //     cleanupDenoiser();
    //     initDenoiser();
    // }
}

void OptixRenderer::render(int samplesPerPixel) {
    if (!initialized) return;

    // Safety check: don't render without a valid scene
    if (params.traversable == 0) {
        std::cerr << "[OptixRenderer] Cannot render: no acceleration structure built" << std::endl;
        return;
    }

    // Clear any previous CUDA errors to prevent error cascade
    cudaError_t prevErr = cudaGetLastError();
    if (prevErr != cudaSuccess) {
        // Previous error detected - try to recover by syncing
        cudaDeviceSynchronize();
        cudaGetLastError(); // Clear the error
    }

    // Update params
    params.framebuffer = d_framebuffer;
    params.width = width;
    params.height = height;
    params.renderWidth = width;  // Full resolution rendering
    params.renderHeight = height;
    params.frame_number = frame_number++;
    params.samples_per_pixel = samplesPerPixel;
    params.useFlowBuffer = false;  // Motion vectors disabled
    params.flowBuffer = nullptr;
    // params.traversable, params.lights set when scene loaded

    cudaError_t copyErr = cudaMemcpy((void*)d_params, &params, sizeof(OptixParams), cudaMemcpyHostToDevice);
    if (copyErr != cudaSuccess) {
        std::cerr << "[OptixRenderer] Failed to copy params: " << cudaGetErrorString(copyErr) << std::endl;
        // Try to recover
        cudaDeviceSynchronize();
        cudaGetLastError();
        return;
    }

    // Launch rays at full resolution
    int launchWidth = width;
    int launchHeight = height;

    OPTIX_CHECK_VOID(optixLaunch(
        pipeline,
        0, // CUDA stream
        d_params,
        sizeof(OptixParams),
        &sbt,
        launchWidth,
        launchHeight,
        1 // depth
    ));

    // Sync and check for errors
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "[OptixRenderer] Post-launch sync error: " << cudaGetErrorString(syncErr) << std::endl;
        cudaGetLastError(); // Clear error for next frame
        return;
    }

    // Apply AI denoiser with temporal upscaling (DLSS-like)
    if (denoiserEnabled && denoiser && d_denoisedBuffer) {
        OptixDenoiserParams denoiserParams = {};
        denoiserParams.hdrIntensity = 0;  // Auto-calculate intensity
        denoiserParams.blendFactor = 0.0f;  // 0 = fully denoised, 1 = original
        denoiserParams.hdrAverageColor = 0;  // Auto-calculate

        // For temporal mode: use previous frame data after first frame
        if (useTemporalUpscale && d_prevOutputBuffer && accumFrameCount > 0) {
            denoiserParams.temporalModeUsePreviousLayers = 1;
        } else {
            denoiserParams.temporalModeUsePreviousLayers = 0;
        }

        // Setup input layer (noisy render at render resolution)
        OptixDenoiserLayer layer = {};
        layer.input.data = reinterpret_cast<CUdeviceptr>(d_framebuffer);
        // Input is at RENDER resolution (half-res for DLSS reconstruction)
        int inputWidth = useTemporalUpscale ? renderWidth : width;
        int inputHeight = useTemporalUpscale ? renderHeight : height;
        layer.input.width = inputWidth;
        layer.input.height = inputHeight;
        // rowStride matches the actual buffer width
        layer.input.rowStrideInBytes = inputWidth * sizeof(float4);
        layer.input.pixelStrideInBytes = sizeof(float4);
        layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // Setup output layer (denoised/upscaled - full resolution)
        layer.output.data = reinterpret_cast<CUdeviceptr>(d_denoisedBuffer);
        layer.output.width = width;
        layer.output.height = height;
        layer.output.rowStrideInBytes = width * sizeof(float4);
        layer.output.pixelStrideInBytes = sizeof(float4);
        layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        // Previous output for temporal stability
        if (useTemporalUpscale && d_prevOutputBuffer) {
            layer.previousOutput.data = reinterpret_cast<CUdeviceptr>(d_prevOutputBuffer);
            layer.previousOutput.width = width;
            layer.previousOutput.height = height;
            layer.previousOutput.rowStrideInBytes = width * sizeof(float4);
            layer.previousOutput.pixelStrideInBytes = sizeof(float4);
            layer.previousOutput.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        }

        // Setup guide layer with optical flow (motion vectors) and internal guide layers
        OptixDenoiserGuideLayer guideLayer = {};
        if (useTemporalUpscale && d_flowBuffer) {
            guideLayer.flow.data = reinterpret_cast<CUdeviceptr>(d_flowBuffer);
            // Flow buffer is at RENDER resolution (one motion vector per rendered pixel)
            guideLayer.flow.width = renderWidth;
            guideLayer.flow.height = renderHeight;
            guideLayer.flow.rowStrideInBytes = renderWidth * sizeof(float2);
            guideLayer.flow.pixelStrideInBytes = sizeof(float2);
            guideLayer.flow.format = OPTIX_PIXEL_FORMAT_FLOAT2;
        }

        // Set internal guide layers (REQUIRED for TEMPORAL_UPSCALE2X)
        if (useTemporalUpscale && d_internalGuideLayer && d_prevInternalGuideLayer && internalGuideLayerSize > 0) {
            // Calculate pixel stride from total size
            size_t pixelStride = internalGuideLayerSize / (width * height);

            guideLayer.outputInternalGuideLayer.data = d_internalGuideLayer;
            guideLayer.outputInternalGuideLayer.width = static_cast<unsigned int>(width);
            guideLayer.outputInternalGuideLayer.height = static_cast<unsigned int>(height);
            guideLayer.outputInternalGuideLayer.pixelStrideInBytes = static_cast<unsigned int>(pixelStride);
            guideLayer.outputInternalGuideLayer.rowStrideInBytes = static_cast<unsigned int>(width * pixelStride);
            guideLayer.outputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;

            guideLayer.previousOutputInternalGuideLayer.data = d_prevInternalGuideLayer;
            guideLayer.previousOutputInternalGuideLayer.width = static_cast<unsigned int>(width);
            guideLayer.previousOutputInternalGuideLayer.height = static_cast<unsigned int>(height);
            guideLayer.previousOutputInternalGuideLayer.pixelStrideInBytes = static_cast<unsigned int>(pixelStride);
            guideLayer.previousOutputInternalGuideLayer.rowStrideInBytes = static_cast<unsigned int>(width * pixelStride);
            guideLayer.previousOutputInternalGuideLayer.format = OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
        }

        OptixResult denoiseResult = optixDenoiserInvoke(
            denoiser,
            0,  // CUDA stream
            &denoiserParams,
            d_denoiserState,
            denoiserStateSize,
            &guideLayer,
            &layer,
            1,  // num layers
            0,  // input offset x
            0,  // input offset y
            d_denoiserScratch,
            denoiserScratchSize
        );

        if (denoiseResult != OPTIX_SUCCESS) {
            std::cerr << "[OptixRenderer] Denoiser invoke failed: " << optixGetErrorName(denoiseResult) << std::endl;
        }

        cudaDeviceSynchronize();

        // Copy current output to previous buffer for next frame's temporal reference
        if (useTemporalUpscale && d_prevOutputBuffer) {
            cudaMemcpy(d_prevOutputBuffer, d_denoisedBuffer, width * height * sizeof(float4), cudaMemcpyDeviceToDevice);
        }

        // Swap internal guide layers for next frame (current becomes previous)
        if (useTemporalUpscale && d_internalGuideLayer && d_prevInternalGuideLayer && internalGuideLayerSize > 0) {
            cudaMemcpy(reinterpret_cast<void*>(d_prevInternalGuideLayer),
                      reinterpret_cast<void*>(d_internalGuideLayer),
                      internalGuideLayerSize,
                      cudaMemcpyDeviceToDevice);
        }

        accumFrameCount++;
    }
}

void OptixRenderer::updateCamera(vec3 eye, vec3 lookat, vec3 up, float fov, float aperture, float focus_dist) {
    // Save previous camera state for motion vectors (Phase 1: gated, won't execute)
    if (useTemporalUpscale) {
        prev_cam_eye = params.cam_eye;
        prev_cam_u = params.cam_u;
        prev_cam_v = params.cam_v;
        prev_cam_w = params.cam_w;
    }

    float aspect = float(width) / float(height);
    float theta = fov * 3.14159265f / 180.0f;
    float h = tan(theta / 2.0f);
    float viewport_height = 2.0f * h * focus_dist;
    float viewport_width = aspect * viewport_height;

    vec3 w = unit_vector(eye - lookat);
    vec3 u = unit_vector(cross(up, w));
    vec3 v = cross(w, u);

    params.cam_eye = make_float3(eye.x(), eye.y(), eye.z());
    // cam_u and cam_v are HALF the viewport dimensions (shader uses (2*u-1) to go from -1 to +1)
    float half_width = viewport_width / 2.0f;
    float half_height = viewport_height / 2.0f;
    params.cam_u = make_float3(u.x() * half_width, u.y() * half_width, u.z() * half_width);
    params.cam_v = make_float3(v.x() * half_height, v.y() * half_height, v.z() * half_height);
    params.cam_w = make_float3(-w.x() * focus_dist, -w.y() * focus_dist, -w.z() * focus_dist);

    // Depth of field parameters
    float lens_radius = aperture / 2.0f;
    params.lens_radius = lens_radius;
    params.defocus_disk_u = make_float3(u.x() * lens_radius, u.y() * lens_radius, u.z() * lens_radius);
    params.defocus_disk_v = make_float3(v.x() * lens_radius, v.y() * lens_radius, v.z() * lens_radius);

    // Reset accumulation when camera changes
    resetAccumulation();
}

void OptixRenderer::cleanup() {
    cleanupDenoiser();

    if (pipeline) optixPipelineDestroy(pipeline);
    if (raygenPG) optixProgramGroupDestroy(raygenPG);
    if (missPG) optixProgramGroupDestroy(missPG);
    if (missShadowPG) optixProgramGroupDestroy(missShadowPG);
    if (hitgroupSpherePG) optixProgramGroupDestroy(hitgroupSpherePG);
    if (hitgroupTrianglePG) optixProgramGroupDestroy(hitgroupTrianglePG);
    if (module) optixModuleDestroy(module);
    if (sphereModule) optixModuleDestroy(sphereModule);
    if (context) optixDeviceContextDestroy(context);

    if (d_raygenRecord) cudaFree((void*)d_raygenRecord);
    if (d_missRecord) cudaFree((void*)d_missRecord);
    if (d_hitgroupRecord) cudaFree((void*)d_hitgroupRecord);
    if (d_params) cudaFree((void*)d_params);
    if (d_sphereGasBuffer) cudaFree((void*)d_sphereGasBuffer);
    if (d_triangleGasBuffer) cudaFree((void*)d_triangleGasBuffer);
    if (d_iasBuffer) cudaFree((void*)d_iasBuffer);
    if (d_instances) cudaFree((void*)d_instances);
    if (d_sphereVertices) cudaFree((void*)d_sphereVertices);
    if (d_sphereRadii) cudaFree((void*)d_sphereRadii);
    if (d_sbtIndices) cudaFree((void*)d_sbtIndices);
    if (d_triangleVertices) cudaFree((void*)d_triangleVertices);
    if (d_triangleIndices) cudaFree((void*)d_triangleIndices);
    if (d_triangleSbtIndices) cudaFree((void*)d_triangleSbtIndices);
    if (d_lights) cudaFree(d_lights);

    freeFramebuffer();

    pipeline = nullptr;
    raygenPG = missPG = missShadowPG = hitgroupSpherePG = hitgroupTrianglePG = nullptr;
    module = nullptr;
    sphereModule = nullptr;
    context = nullptr;
    d_sphereGasBuffer = 0;
    d_triangleGasBuffer = 0;
    d_iasBuffer = 0;
    d_instances = 0;
    d_sphereVertices = 0;
    d_sphereRadii = 0;
    d_sbtIndices = 0;
    d_triangleVertices = 0;
    d_triangleIndices = 0;
    d_triangleSbtIndices = 0;
    numTriangles = 0;
    d_lights = nullptr;
    num_lights = 0;
    initialized = false;
}

bool OptixRenderer::initDenoiser() {
    // Use TEMPORAL_UPSCALE2X for DLSS-like ray reconstruction
    // Render at half resolution, AI upscales to full with temporal stability
    renderWidth = width / 2;
    renderHeight = height / 2;

    // Create denoiser with temporal upscaling
    OptixDenoiserOptions denoiserOptions = {};
    denoiserOptions.guideAlbedo = 0;  // We don't have albedo AOV
    denoiserOptions.guideNormal = 0;  // We don't have normal AOV
    denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

    OptixDenoiserModelKind modelKind = useTemporalUpscale ?
        OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X :
        OPTIX_DENOISER_MODEL_KIND_AOV;

    OptixResult result = optixDenoiserCreate(context, modelKind, &denoiserOptions, &denoiser);
    if (result != OPTIX_SUCCESS) {
        std::cerr << "[OptixRenderer] Failed to create denoiser: " << optixGetErrorName(result) << std::endl;
        // Fallback to simple AOV mode
        useTemporalUpscale = false;
        renderWidth = width;
        renderHeight = height;
        result = optixDenoiserCreate(context, OPTIX_DENOISER_MODEL_KIND_AOV, &denoiserOptions, &denoiser);
        if (result != OPTIX_SUCCESS) {
            denoiserEnabled = false;
            return false;
        }
    }

    // Compute memory requirements (use output dimensions for upscale mode)
    OptixDenoiserSizes denoiserSizes;
    result = optixDenoiserComputeMemoryResources(denoiser, width, height, &denoiserSizes);
    if (result != OPTIX_SUCCESS) {
        std::cerr << "[OptixRenderer] Failed to compute denoiser memory: " << optixGetErrorName(result) << std::endl;
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
        denoiserEnabled = false;
        return false;
    }

    denoiserStateSize = denoiserSizes.stateSizeInBytes;
    denoiserScratchSize = denoiserSizes.withoutOverlapScratchSizeInBytes;

    // Get internal guide layer size for temporal mode
    if (useTemporalUpscale) {
        internalGuideLayerSize = denoiserSizes.internalGuideLayerPixelSizeInBytes * width * height;
        std::cout << "[OptixRenderer] Internal guide layer size: " << internalGuideLayerSize << " bytes" << std::endl;
    }

    // Allocate denoiser state and scratch buffers
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_denoiserState), denoiserStateSize);
    if (err != cudaSuccess) {
        std::cerr << "[OptixRenderer] Failed to allocate denoiser state" << std::endl;
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
        denoiserEnabled = false;
        return false;
    }

    err = cudaMalloc(reinterpret_cast<void**>(&d_denoiserScratch), denoiserScratchSize);
    if (err != cudaSuccess) {
        std::cerr << "[OptixRenderer] Failed to allocate denoiser scratch" << std::endl;
        cudaFree(reinterpret_cast<void*>(d_denoiserState));
        d_denoiserState = 0;
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
        denoiserEnabled = false;
        return false;
    }

    // Allocate denoised output buffer (full resolution)
    err = cudaMalloc(reinterpret_cast<void**>(&d_denoisedBuffer), width * height * sizeof(float4));
    if (err != cudaSuccess) {
        std::cerr << "[OptixRenderer] Failed to allocate denoised buffer" << std::endl;
        cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
        cudaFree(reinterpret_cast<void*>(d_denoiserState));
        d_denoiserScratch = 0;
        d_denoiserState = 0;
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
        denoiserEnabled = false;
        return false;
    }

    // Allocate previous output buffer for temporal mode (full resolution)
    if (useTemporalUpscale) {
        err = cudaMalloc(reinterpret_cast<void**>(&d_prevOutputBuffer), width * height * sizeof(float4));
        if (err != cudaSuccess) {
            std::cerr << "[OptixRenderer] Failed to allocate previous output buffer" << std::endl;
            // Continue without temporal - just won't have prev frame data
        } else {
            cudaMemset(d_prevOutputBuffer, 0, width * height * sizeof(float4));
        }

        // Allocate motion vector buffer at RENDER resolution (one vector per rendered pixel)
        err = cudaMalloc(reinterpret_cast<void**>(&d_flowBuffer), renderWidth * renderHeight * sizeof(float2));
        if (err != cudaSuccess) {
            std::cerr << "[OptixRenderer] Failed to allocate flow buffer" << std::endl;
            // Continue without flow - denoiser will work without motion vectors
        } else {
            cudaMemset(d_flowBuffer, 0, renderWidth * renderHeight * sizeof(float2));
            params.flowBuffer = d_flowBuffer;  // Set params pointer
        }

        // Allocate internal guide layer buffers (required for TEMPORAL_UPSCALE2X)
        if (internalGuideLayerSize > 0) {
            err = cudaMalloc(reinterpret_cast<void**>(&d_internalGuideLayer), internalGuideLayerSize);
            if (err != cudaSuccess) {
                std::cerr << "[OptixRenderer] Failed to allocate internal guide layer" << std::endl;
                return false;
            }
            cudaMemset(reinterpret_cast<void*>(d_internalGuideLayer), 0, internalGuideLayerSize);

            err = cudaMalloc(reinterpret_cast<void**>(&d_prevInternalGuideLayer), internalGuideLayerSize);
            if (err != cudaSuccess) {
                std::cerr << "[OptixRenderer] Failed to allocate previous internal guide layer" << std::endl;
                cudaFree(reinterpret_cast<void*>(d_internalGuideLayer));
                d_internalGuideLayer = 0;
                return false;
            }
            cudaMemset(reinterpret_cast<void*>(d_prevInternalGuideLayer), 0, internalGuideLayerSize);

            std::cout << "[OptixRenderer] Allocated internal guide layers for temporal mode" << std::endl;
        }
    }

    // Setup denoiser with output dimensions
    result = optixDenoiserSetup(
        denoiser,
        0,  // CUDA stream
        width,   // Output width (full res)
        height,  // Output height (full res)
        d_denoiserState,
        denoiserStateSize,
        d_denoiserScratch,
        denoiserScratchSize
    );
    if (result != OPTIX_SUCCESS) {
        std::cerr << "[OptixRenderer] Failed to setup denoiser: " << optixGetErrorName(result) << std::endl;
        cleanupDenoiser();
        denoiserEnabled = false;
        return false;
    }

    if (useTemporalUpscale) {
        std::cout << "[OptixRenderer] AI Denoiser: TEMPORAL_UPSCALE2X (DLSS-like Ray Reconstruction)" << std::endl;
        std::cout << "[OptixRenderer] Render: " << renderWidth << "x" << renderHeight << " -> Upscale: " << width << "x" << height << std::endl;
    } else {
        std::cout << "[OptixRenderer] AI Denoiser: AOV mode" << std::endl;
    }
    std::cout << "[OptixRenderer] Tensor Cores: Used for AI denoising/upscaling" << std::endl;
    denoiserEnabled = true;
    return true;
}

void OptixRenderer::cleanupDenoiser() {
    if (d_denoisedBuffer) {
        cudaFree(d_denoisedBuffer);
        d_denoisedBuffer = nullptr;
    }
    if (d_prevOutputBuffer) {
        cudaFree(d_prevOutputBuffer);
        d_prevOutputBuffer = nullptr;
    }
    if (d_flowBuffer) {
        cudaFree(d_flowBuffer);
        d_flowBuffer = nullptr;
    }
    if (d_internalGuideLayer) {
        cudaFree(reinterpret_cast<void*>(d_internalGuideLayer));
        d_internalGuideLayer = 0;
    }
    if (d_prevInternalGuideLayer) {
        cudaFree(reinterpret_cast<void*>(d_prevInternalGuideLayer));
        d_prevInternalGuideLayer = 0;
    }
    if (d_denoiserScratch) {
        cudaFree(reinterpret_cast<void*>(d_denoiserScratch));
        d_denoiserScratch = 0;
    }
    if (d_denoiserState) {
        cudaFree(reinterpret_cast<void*>(d_denoiserState));
        d_denoiserState = 0;
    }
    if (denoiser) {
        optixDenoiserDestroy(denoiser);
        denoiser = nullptr;
    }
}

// Sphere-only buildAccel - delegates to the combined version with empty triangles
bool OptixRenderer::buildAccel(const std::vector<SphereData>& spheres) {
    std::vector<TriangleData> emptyTriangles;
    return buildAccel(spheres, emptyTriangles);
}

// Legacy buildAccel without materials (creates default red material)
bool OptixRenderer::buildAccel(const std::vector<float3>& centers, const std::vector<float>& radii) {
    if (centers.empty() || centers.size() != radii.size()) {
        std::cerr << "[OptixRenderer] buildAccel: Invalid input" << std::endl;
        return false;
    }

    // Convert to SphereData with default red lambertian material
    std::vector<SphereData> spheres;
    spheres.reserve(centers.size());
    for (size_t i = 0; i < centers.size(); i++) {
        SphereData s;
        s.center = vec3(centers[i].x, centers[i].y, centers[i].z);
        s.radius = radii[i];
        s.material_type = 0;  // Lambertian
        s.albedo = vec3(0.8f, 0.3f, 0.3f);  // Red
        s.fuzz_or_ior = 0.0f;
        spheres.push_back(s);
    }

    return buildAccel(spheres);
}

void OptixRenderer::setLights(const std::vector<LightData>& lights) {
    // Free previous light data
    if (d_lights) {
        cudaFree(d_lights);
        d_lights = nullptr;
    }

    num_lights = static_cast<int>(lights.size());
    if (num_lights == 0) {
        params.lights = nullptr;
        params.num_lights = 0;
        return;
    }

    // Convert LightData to OptixLightData
    std::vector<OptixLightData> optixLights(num_lights);
    for (int i = 0; i < num_lights; i++) {
        optixLights[i].type = lights[i].type;
        optixLights[i].position = make_float3(
            lights[i].position.x(), lights[i].position.y(), lights[i].position.z());
        optixLights[i].direction = make_float3(
            lights[i].direction.x(), lights[i].direction.y(), lights[i].direction.z());
        optixLights[i].color = make_float3(
            lights[i].color.x(), lights[i].color.y(), lights[i].color.z());
        optixLights[i].intensity = lights[i].intensity;
        optixLights[i].radius = lights[i].radius;
    }

    // Upload to GPU
    cudaMalloc((void**)&d_lights, num_lights * sizeof(OptixLightData));
    cudaMemcpy(d_lights, optixLights.data(), num_lights * sizeof(OptixLightData), cudaMemcpyHostToDevice);

    params.lights = d_lights;
    params.num_lights = num_lights;

    std::cout << "[OptixRenderer] Set " << num_lights << " lights" << std::endl;
}

void OptixRenderer::setMaterials(const std::vector<MaterialData>& materials) {
    // Update materials in the cached sphere data and rebuild GAS
    // For now, this requires rebuilding the acceleration structure
    // A more optimal implementation would use SBT updates

    // Convert MaterialData to update our cached spheres
    for (size_t i = 0; i < materials.size() && i < cachedSpheres.size(); ++i) {
        cachedSpheres[i].material_type = materials[i].type;
        cachedSpheres[i].albedo = vec3(materials[i].albedo.x(), materials[i].albedo.y(), materials[i].albedo.z());
        cachedSpheres[i].fuzz_or_ior = (materials[i].type == 1) ? materials[i].fuzz : materials[i].ior;
    }

    // Rebuild acceleration structure with updated materials
    if (!cachedSpheres.empty() || !cachedTriangles.empty()) {
        buildAccel(cachedSpheres, cachedTriangles);
    }
}

void OptixRenderer::setMeshMaterial(int meshIndex, int materialType, vec3 albedo, float fuzz_or_ior) {
    std::cout << "[OptixRenderer::setMeshMaterial] Called with meshIndex=" << meshIndex
              << ", cachedMeshes.size()=" << cachedMeshes.size()
              << ", cachedTriangles.size()=" << cachedTriangles.size() << std::endl;

    if (meshIndex < 0 || meshIndex >= static_cast<int>(cachedMeshes.size())) {
        std::cerr << "[OptixRenderer::setMeshMaterial] Invalid mesh index: " << meshIndex
                  << " (cachedMeshes has " << cachedMeshes.size() << " entries)" << std::endl;
        return;
    }
    MeshData& mesh = cachedMeshes[meshIndex];
    std::cout << "[OptixRenderer::setMeshMaterial] Mesh '" << mesh.name << "' triangleStart="
              << mesh.triangleStart << " triangleCount=" << mesh.triangleCount << std::endl;

    mesh.material_type = materialType;
    mesh.albedo = albedo;
    mesh.fuzz_or_ior = fuzz_or_ior;
    // Update triangles belonging to this mesh
    int updatedCount = 0;
    for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; ++i) {
        if (i >= 0 && i < static_cast<int>(cachedTriangles.size())) {
            cachedTriangles[i].material_type = materialType;
            cachedTriangles[i].albedo = albedo;
            cachedTriangles[i].fuzz_or_ior = fuzz_or_ior;
            updatedCount++;
        }
    }
    std::cout << "[OptixRenderer::setMeshMaterial] Updated " << updatedCount << " triangles" << std::endl;

    // Rebuild with updated materials
    buildAccel(cachedSpheres, cachedTriangles);
    std::cout << "[OptixRenderer::setMeshMaterial] Rebuild complete for mesh " << meshIndex << std::endl;
}

// Build acceleration structure with spheres, triangles, and mesh metadata
bool OptixRenderer::buildAccel(const std::vector<SphereData>& spheres, const std::vector<TriangleData>& triangles,
                                const std::vector<MeshData>& meshes) {
    std::cout << "[OptixRenderer::buildAccel] 3-arg version called with " << meshes.size() << " meshes" << std::endl;
    for (size_t i = 0; i < meshes.size(); i++) {
        std::cout << "  Mesh[" << i << "]: '" << meshes[i].name << "' triStart=" << meshes[i].triangleStart
                  << " triCount=" << meshes[i].triangleCount << std::endl;
    }
    cachedMeshes = meshes;  // Store mesh metadata for material updates
    return buildAccel(spheres, triangles);  // Call existing implementation
}

// Build acceleration structure with both spheres and triangles
bool OptixRenderer::buildAccel(const std::vector<SphereData>& spheres, const std::vector<TriangleData>& triangles) {
    std::cout << "[OptixRenderer::buildAccel] 2-arg version called, cachedMeshes.size()=" << cachedMeshes.size() << std::endl;
    if (spheres.empty() && triangles.empty()) {
        std::cerr << "[OptixRenderer] buildAccel: Empty scene" << std::endl;
        return false;
    }

    // Cache scene data for material updates
    cachedSpheres = spheres;
    cachedTriangles = triangles;

    // Free previous resources
    if (d_sphereGasBuffer) { cudaFree((void*)d_sphereGasBuffer); d_sphereGasBuffer = 0; }
    if (d_triangleGasBuffer) { cudaFree((void*)d_triangleGasBuffer); d_triangleGasBuffer = 0; }
    if (d_iasBuffer) { cudaFree((void*)d_iasBuffer); d_iasBuffer = 0; }
    if (d_instances) { cudaFree((void*)d_instances); d_instances = 0; }
    if (d_sphereVertices) { cudaFree((void*)d_sphereVertices); d_sphereVertices = 0; }
    if (d_sphereRadii) { cudaFree((void*)d_sphereRadii); d_sphereRadii = 0; }
    if (d_sbtIndices) { cudaFree((void*)d_sbtIndices); d_sbtIndices = 0; }
    if (d_triangleVertices) { cudaFree((void*)d_triangleVertices); d_triangleVertices = 0; }
    if (d_triangleSbtIndices) { cudaFree((void*)d_triangleSbtIndices); d_triangleSbtIndices = 0; }
    if (d_hitgroupRecord) { cudaFree((void*)d_hitgroupRecord); d_hitgroupRecord = 0; }

    numSpheres = spheres.size();
    numTriangles = triangles.size();
    sphereGasHandle = 0;
    triangleGasHandle = 0;

    std::cout << "[OptixRenderer] Building separate GAS for " << numSpheres << " spheres and "
              << numTriangles << " triangles (IAS approach)..." << std::endl;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // === BUILD SPHERE GAS ===
    if (numSpheres > 0) {
        std::vector<float3> centers;
        std::vector<float> radii;
        centers.reserve(numSpheres);
        radii.reserve(numSpheres);
        for (const auto& s : spheres) {
            centers.push_back(make_float3(s.center.x(), s.center.y(), s.center.z()));
            radii.push_back(s.radius);
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphereVertices), numSpheres * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sphereVertices), centers.data(), numSpheres * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphereRadii), numSpheres * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sphereRadii), radii.data(), numSpheres * sizeof(float), cudaMemcpyHostToDevice));

        std::vector<uint32_t> sphereSbtIndices(numSpheres);
        for (size_t i = 0; i < numSpheres; i++) sphereSbtIndices[i] = static_cast<uint32_t>(i);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbtIndices), numSpheres * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_sbtIndices), sphereSbtIndices.data(), numSpheres * sizeof(uint32_t), cudaMemcpyHostToDevice));

        std::vector<uint32_t> sphereFlags(numSpheres, OPTIX_GEOMETRY_FLAG_NONE);

        OptixBuildInput sphereInput = {};
        sphereInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        sphereInput.sphereArray.vertexBuffers = &d_sphereVertices;
        sphereInput.sphereArray.numVertices = static_cast<unsigned int>(numSpheres);
        sphereInput.sphereArray.radiusBuffers = &d_sphereRadii;
        sphereInput.sphereArray.flags = sphereFlags.data();
        sphereInput.sphereArray.numSbtRecords = static_cast<unsigned int>(numSpheres);
        sphereInput.sphereArray.sbtIndexOffsetBuffer = d_sbtIndices;
        sphereInput.sphereArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        sphereInput.sphereArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &sphereInput, 1, &gasBufferSizes));

        CUdeviceptr d_tempBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sphereGasBuffer), gasBufferSizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &sphereInput, 1,
            d_tempBuffer, gasBufferSizes.tempSizeInBytes,
            d_sphereGasBuffer, gasBufferSizes.outputSizeInBytes,
            &sphereGasHandle, nullptr, 0));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
        std::cout << "[OptixRenderer] Sphere GAS built, handle: " << sphereGasHandle << std::endl;
    }

    // === BUILD TRIANGLE GAS ===
    if (numTriangles > 0) {
        std::vector<float3> triVerts;
        triVerts.reserve(numTriangles * 3);
        for (const auto& t : triangles) {
            triVerts.push_back(make_float3(t.v0.x(), t.v0.y(), t.v0.z()));
            triVerts.push_back(make_float3(t.v1.x(), t.v1.y(), t.v1.z()));
            triVerts.push_back(make_float3(t.v2.x(), t.v2.y(), t.v2.z()));
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangleVertices), triVerts.size() * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_triangleVertices), triVerts.data(), triVerts.size() * sizeof(float3), cudaMemcpyHostToDevice));

        std::vector<uint32_t> triSbtIndices(numTriangles);
        for (size_t i = 0; i < numTriangles; i++) triSbtIndices[i] = static_cast<uint32_t>(numSpheres + i);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangleSbtIndices), numTriangles * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_triangleSbtIndices), triSbtIndices.data(), numTriangles * sizeof(uint32_t), cudaMemcpyHostToDevice));

        std::vector<uint32_t> triangleFlags(numTriangles, OPTIX_GEOMETRY_FLAG_NONE);

        OptixBuildInput triangleInput = {};
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput.triangleArray.numVertices = static_cast<unsigned int>(triVerts.size());
        triangleInput.triangleArray.vertexBuffers = &d_triangleVertices;
        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
        triangleInput.triangleArray.indexStrideInBytes = 0;
        triangleInput.triangleArray.numIndexTriplets = 0;
        triangleInput.triangleArray.indexBuffer = 0;
        triangleInput.triangleArray.flags = triangleFlags.data();
        triangleInput.triangleArray.numSbtRecords = static_cast<unsigned int>(numTriangles);
        triangleInput.triangleArray.sbtIndexOffsetBuffer = d_triangleSbtIndices;
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &triangleInput, 1, &gasBufferSizes));

        CUdeviceptr d_tempBuffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), gasBufferSizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangleGasBuffer), gasBufferSizes.outputSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(context, 0, &accelOptions, &triangleInput, 1,
            d_tempBuffer, gasBufferSizes.tempSizeInBytes,
            d_triangleGasBuffer, gasBufferSizes.outputSizeInBytes,
            &triangleGasHandle, nullptr, 0));

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));
        std::cout << "[OptixRenderer] Triangle GAS built, handle: " << triangleGasHandle << std::endl;
    }

    // === BUILD IAS (Instance Acceleration Structure) ===
    std::vector<OptixInstance> instances;

    // Identity transform matrix (row-major 3x4)
    float identity[12] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };

    if (sphereGasHandle) {
        OptixInstance sphereInstance = {};
        memcpy(sphereInstance.transform, identity, sizeof(identity));
        sphereInstance.instanceId = 0;
        sphereInstance.sbtOffset = 0;  // Sphere SBT records start at 0
        sphereInstance.visibilityMask = 255;
        sphereInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        sphereInstance.traversableHandle = sphereGasHandle;
        instances.push_back(sphereInstance);
    }

    if (triangleGasHandle) {
        OptixInstance triangleInstance = {};
        memcpy(triangleInstance.transform, identity, sizeof(identity));
        triangleInstance.instanceId = 1;
        triangleInstance.sbtOffset = static_cast<unsigned int>(numSpheres);  // Triangle SBT records start after spheres
        triangleInstance.visibilityMask = 255;
        triangleInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        triangleInstance.traversableHandle = triangleGasHandle;
        instances.push_back(triangleInstance);
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_instances), instances.size() * sizeof(OptixInstance)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances), instances.data(), instances.size() * sizeof(OptixInstance), cudaMemcpyHostToDevice));

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = d_instances;
    instanceInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    OptixAccelBuildOptions iasOptions = {};
    iasOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    iasOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &iasOptions, &instanceInput, 1, &iasBufferSizes));

    CUdeviceptr d_tempBuffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tempBuffer), iasBufferSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_iasBuffer), iasBufferSizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &iasOptions, &instanceInput, 1,
        d_tempBuffer, iasBufferSizes.tempSizeInBytes,
        d_iasBuffer, iasBufferSizes.outputSizeInBytes,
        &iasHandle, nullptr, 0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_tempBuffer)));

    params.traversable = iasHandle;
    std::cout << "[OptixRenderer] IAS built with " << instances.size() << " instances, handle: " << iasHandle << std::endl;

    // === BUILD SBT HITGROUP RECORDS ===
    size_t totalRecords = numSpheres + numTriangles;
    std::vector<HitGroupRecord> hitRecords(totalRecords);

    // Sphere hit records
    for (size_t i = 0; i < numSpheres; i++) {
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupSpherePG, &hitRecords[i]));
        hitRecords[i].material.type = spheres[i].material_type;
        hitRecords[i].material.albedo = make_float3(spheres[i].albedo.x(), spheres[i].albedo.y(), spheres[i].albedo.z());
        hitRecords[i].material.fuzz_or_ior = spheres[i].fuzz_or_ior;
    }

    // Triangle hit records
    for (size_t i = 0; i < numTriangles; i++) {
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupTrianglePG, &hitRecords[numSpheres + i]));
        hitRecords[numSpheres + i].material.type = triangles[i].material_type;
        hitRecords[numSpheres + i].material.albedo = make_float3(triangles[i].albedo.x(), triangles[i].albedo.y(), triangles[i].albedo.z());
        hitRecords[numSpheres + i].material.fuzz_or_ior = triangles[i].fuzz_or_ior;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_hitgroupRecord, totalRecords * sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy((void*)d_hitgroupRecord, hitRecords.data(), totalRecords * sizeof(HitGroupRecord), cudaMemcpyHostToDevice));

    sbt.hitgroupRecordBase = d_hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(totalRecords);

    std::cout << "[OptixRenderer] Accel built successfully: " << numSpheres << " spheres + "
              << numTriangles << " triangles via IAS" << std::endl;
    return true;
}

