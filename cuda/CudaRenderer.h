#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "vec3.h"
#include "bvh.h"

// Suppress warnings from NVIDIA's curand_kernel.h
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100)  // unreferenced formal parameter
#endif
#include <curand_kernel.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <vector>
#include <string>

// CPU-side sphere data for BVH building
struct SphereData {
    vec3 center;
    float radius;
    int material_type;  // 0=lambertian, 1=metal, 2=dielectric
    vec3 albedo;
    float fuzz_or_ior;  // fuzz for metal, ior for dielectric
    std::string name;   // Object name from scene
};

// CPU-side triangle data for BVH building
struct TriangleData {
    vec3 v0, v1, v2;           // Vertices
    vec3 n0, n1, n2;           // Per-vertex normals
    bool use_vertex_normals;
    int material_type;         // 0=lambertian, 1=metal, 2=dielectric
    vec3 albedo;
    float fuzz_or_ior;
};

// Light data for GPU
struct LightData {
    int type;           // 0=point, 1=distant, 2=sphere, 3=rect, 4=spot
    vec3 position;
    vec3 direction;     // For distant lights and spotlights
    vec3 color;
    float intensity;
    float radius;       // For sphere lights
    float cone_angle;   // For spotlights (in radians, half-angle)
    float falloff;      // For spotlights (softness of edge)
};

// Material data for dynamic updates
struct MaterialData {
    int type;           // 0=lambertian, 1=metal, 2=dielectric
    vec3 albedo;
    float fuzz;
    float ior;
};

// Mesh metadata for tracking mesh objects (groups of triangles)
struct MeshData {
    std::string name;
    int triangleStart;  // Index of first triangle in h_triangles
    int triangleCount;  // Number of triangles in this mesh
    int material_type;
    vec3 albedo;
    float fuzz_or_ior;
};

// CPU-side BVH node for building
struct BVHBuildNode {
    aabb bounds;
    int left;           // Left child index or -1
    int right;          // Right child index or -1
    int prim_idx;       // Primitive index if leaf, -1 otherwise
    int is_leaf;        // Use int instead of bool for GPU alignment
};

class CudaRenderer {
public:
    CudaRenderer();
    ~CudaRenderer();

    // Initialize CUDA resources for given dimensions
    bool initialize(int width, int height);

    // Resize frame buffer (reallocates VRAM)
    void resize(int width, int height);

    // Render the scene
    void render(int samplesPerPixel);

    // Clean up all GPU resources
    void cleanup();

    // Update camera parameters on GPU
    void updateCamera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aperture, float focus_dist);

    // Scene I/O
    bool exportSceneToUSDA(const std::string& filename);
    bool loadSceneFromUSDA(const std::string& filename);

    // Get pointer to frame buffer (device memory, vec3 per pixel)
    vec3* getFramebuffer() const { return d_framebuffer; }

    // Get frame buffer dimensions
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    // Get sphere data for sharing with OptiX
    const std::vector<SphereData>& getSpheres() const { return h_spheres; }

    // Get triangle data for sharing with OptiX
    const std::vector<TriangleData>& getTriangles() const { return h_triangles; }

    // Get mesh metadata for material controls
    const std::vector<MeshData>& getMeshes() const { return h_meshes; }

    // Get light data for sharing with OptiX
    const std::vector<LightData>& getLights() const { return h_lights; }

    // Set light data (for runtime updates)
    void setLights(const std::vector<LightData>& lights);

    // Set material data (for runtime updates)
    void setMaterials(const std::vector<MaterialData>& materials);

    // Update a single mesh's material (updates all its triangles)
    void setMeshMaterial(int meshIndex, int materialType, vec3 albedo, float fuzz_or_ior);

private:
    // Helper to rebuild GPU world from existing CPU data (without calling generateSphereData)
    void rebuildGPUWorld();
    // Device pointers
    vec3* d_framebuffer;
    curandState* d_rand_state;
    curandState* d_rand_state2;  // for world creation

    // Scene pointers
    void** d_list;   // hitable** - array of sphere pointers
    void** d_world;  // hitable** - points to BVH
    void** d_camera; // camera**

    // BVH device memory
    BVHFlatNode* d_bvh_nodes;  // Flat BVH node array on GPU

    // Light device memory
    LightData* d_lights;
    int num_lights;

    int width, height;
    int num_hitables;
    int num_bvh_nodes;
    int num_spheres_in_scene;    // Track sphere count for cleanup
    int num_triangles_in_scene;  // Track triangle count for cleanup
    bool initialized;

    // Block/thread configuration
    int tx, ty;

    // Max depth configuration (can be changed at runtime, unlike OptiX)
    int maxDepth;

    // Sky/background configuration
    bool disableSky;

public:
    void setMaxDepth(int depth) { maxDepth = depth; }
    int getMaxDepth() const { return maxDepth; }
    void setDisableSky(bool disable) { disableSky = disable; }
    bool getDisableSky() const { return disableSky; }

private:
    // CPU-side data for BVH building
    std::vector<SphereData> h_spheres;
    std::vector<TriangleData> h_triangles;
    std::vector<MeshData> h_meshes;  // Mesh metadata for material controls
    std::vector<BVHBuildNode> h_bvh_nodes;
    std::vector<LightData> h_lights;

    // Helper methods
    void allocateFramebuffer();
    void freeFramebuffer();
    void createWorld();
    void freeWorld();

    // BVH building (CPU-side)
    void generateSphereData();
    int buildBVH(std::vector<int>& prim_indices, int start, int end);
    void copyBVHToDevice();

    // Primitive helpers for BVH (handles both spheres and triangles)
    aabb getPrimitiveBounds(int prim_idx) const;
    vec3 getPrimitiveCentroid(int prim_idx) const;
};

#endif