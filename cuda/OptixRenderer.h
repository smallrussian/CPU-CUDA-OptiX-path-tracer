#ifndef OPTIX_RENDERER_H
#define OPTIX_RENDERER_H

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "vec3.h"
#include "CudaRenderer.h"  // For SphereData and LightData

// Material data for OptiX shading
struct OptixMaterialData {
    int type;           // 0=lambertian, 1=metal, 2=dielectric
    float3 albedo;
    float fuzz_or_ior;  // fuzz for metal, ior for dielectric
};

// Light data for OptiX (matches CudaRenderer's LightData)
struct OptixLightData {
    int type;           // 0=point, 1=distant, 2=sphere, 3=rect
    float3 position;
    float3 direction;   // For distant lights
    float3 color;
    float intensity;
    float radius;       // For sphere lights
};

// Launch parameters passed to OptiX programs
struct OptixParams {
    float4* framebuffer;
    float2* flowBuffer;     // Motion vectors for temporal denoising (pixel displacement)
    int width;
    int height;
    int renderWidth;   // Actual ray trace resolution (half-res for upscaling)
    int renderHeight;
    OptixTraversableHandle traversable;

    // Camera (current frame)
    float3 cam_eye;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;

    // Previous frame camera (for motion vectors)
    float3 prev_cam_eye;
    float3 prev_cam_u;
    float3 prev_cam_v;
    float3 prev_cam_w;

    // Depth of field
    float3 defocus_disk_u;
    float3 defocus_disk_v;
    float lens_radius;

    // Lights
    OptixLightData* lights;
    int num_lights;

    // Random seed for sampling
    unsigned int frame_number;

    // Samples per pixel (from viewport settings)
    int samples_per_pixel;

    // Flow buffer enabled flag (prevents writing to null buffer)
    bool useFlowBuffer;
};

// SBT record types
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RayGenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    float3 bg_color;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OptixMaterialData material;
};

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();

    bool initialize(int width, int height);
    void resize(int width, int height);
    void render(int samplesPerPixel = 1);
    void updateCamera(vec3 eye, vec3 lookat, vec3 up, float fov, float aperture = 0.0f, float focus_dist = 10.0f);
    float4* getFramebuffer() const {
        return (denoiserEnabled && d_denoisedBuffer) ? d_denoisedBuffer : d_framebuffer;
    }
    void cleanup();
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Denoiser control
    void setDenoiserEnabled(bool enabled) { denoiserEnabled = enabled; }
    bool isDenoiserEnabled() const { return denoiserEnabled; }
    void toggleDenoiser() { denoiserEnabled = !denoiserEnabled; }

    // Frame accumulation control
    void resetAccumulation() { accumFrameCount = 0; }
    unsigned int getAccumFrameCount() const { return accumFrameCount; }

    // Scene building with materials
    bool buildAccel(const std::vector<float3>& centers, const std::vector<float>& radii);
    bool buildAccel(const std::vector<SphereData>& spheres);  // Spheres only
    bool buildAccel(const std::vector<SphereData>& spheres, const std::vector<TriangleData>& triangles);  // Full scene
    bool buildAccel(const std::vector<SphereData>& spheres, const std::vector<TriangleData>& triangles,
                    const std::vector<MeshData>& meshes);  // Full scene with mesh metadata
    const std::vector<MeshData>& getMeshes() const { return cachedMeshes; }

    // Lighting
    void setLights(const std::vector<LightData>& lights);

    // Material updates
    void setMaterials(const std::vector<MaterialData>& materials);

    void setMeshMaterial(int meshIndex, int materialType, vec3 albedo, float fuzz_or_ior);

private:
    // OptiX handles
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};

    // Program groups
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup missShadowPG = nullptr;  // Shadow miss program
    OptixProgramGroup hitgroupSpherePG = nullptr;  // Sphere hit group
    OptixProgramGroup hitgroupTrianglePG = nullptr;  // Triangle hit group

    // Device memory
    float4* d_framebuffer = nullptr;
    CUdeviceptr d_params = 0;
    CUdeviceptr d_raygenRecord = 0;
    CUdeviceptr d_missRecord = 0;
    CUdeviceptr d_hitgroupRecord = 0;

    // Acceleration structures - separate GAS for spheres and triangles, combined via IAS
    CUdeviceptr d_sphereGasBuffer = 0;      // GAS for spheres
    CUdeviceptr d_triangleGasBuffer = 0;    // GAS for triangles
    CUdeviceptr d_iasBuffer = 0;            // IAS combining both GAS
    OptixTraversableHandle sphereGasHandle = 0;
    OptixTraversableHandle triangleGasHandle = 0;
    OptixTraversableHandle iasHandle = 0;
    CUdeviceptr d_instances = 0;            // Instance buffer for IAS

    // Sphere data (must persist for GAS traversal)
    CUdeviceptr d_sphereVertices = 0;
    CUdeviceptr d_sphereRadii = 0;
    CUdeviceptr d_sbtIndices = 0;
    OptixModule sphereModule = nullptr;
    size_t numSpheres = 0;

    // Triangle data
    CUdeviceptr d_triangleVertices = 0;
    CUdeviceptr d_triangleIndices = 0;
    CUdeviceptr d_triangleSbtIndices = 0;
    size_t numTriangles = 0;

    // Cached scene data for material updates
    std::vector<SphereData> cachedSpheres;
    std::vector<TriangleData> cachedTriangles;
    std::vector<MeshData> cachedMeshes;  // Mesh metadata for material controls

    // Light device memory
    OptixLightData* d_lights = nullptr;
    int num_lights = 0;

    // Frame counter for random seeds
    unsigned int frame_number = 0;

    // Launch parameters
    OptixParams params = {};

    int width = 0;
    int height = 0;
    bool initialized = false;

    // Denoiser
    OptixDenoiser denoiser = nullptr;
    CUdeviceptr d_denoiserState = 0;
    CUdeviceptr d_denoiserScratch = 0;
    float4* d_denoisedBuffer = nullptr;
    size_t denoiserStateSize = 0;
    size_t denoiserScratchSize = 0;
    bool denoiserEnabled = false;  // DISABLED: No denoiser, just raw rendering with motion vectors

    // Frame accumulation for temporal stability
    float4* d_accumBuffer = nullptr;
    unsigned int accumFrameCount = 0;
    bool cameraMovedThisFrame = false;

    // Temporal denoiser buffers
    float4* d_prevOutputBuffer = nullptr;         // Previous frame output for temporal
    float2* d_flowBuffer = nullptr;                // Motion vectors for temporal upscaling
    CUdeviceptr d_internalGuideLayer = 0;          // Current frame internal guide layer
    CUdeviceptr d_prevInternalGuideLayer = 0;      // Previous frame internal guide layer
    size_t internalGuideLayerSize = 0;             // Size of internal guide layer buffers
    int renderWidth = 0;
    int renderHeight = 0;
    bool useTemporalUpscale = false;  // DISABLED: Render at full-res, denoiser off, but keep motion vectors

    // Previous frame camera for motion vector computation
    float3 prev_cam_eye = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_cam_u = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_cam_v = make_float3(0.0f, 0.0f, 0.0f);
    float3 prev_cam_w = make_float3(0.0f, 0.0f, 0.0f);

    // Helper methods
    bool initContext();
    bool loadPTX(std::string& ptxCode);
    bool createModule(const std::string& ptxCode);
    bool createProgramGroups();
    bool createPipeline();
    bool createSBT();
    void allocateFramebuffer();
    void freeFramebuffer();
    bool initDenoiser();
    void cleanupDenoiser();
};

#endif
