#include "CudaRenderer.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <map>

#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(99);
    }
}

// CUDA KERNELS

// Get sky color with optional sun (or black if disabled)
__device__ vec3 get_sky_color(const vec3& direction, LightData* lights, int num_lights, bool disable_sky) {
    // If sky is disabled, return black
    if (disable_sky) {
        return vec3(0.0f, 0.0f, 0.0f);
    }

    vec3 unit_direction = unit_vector(direction);
    float t = 0.5f * (unit_direction.y() + 1.0f);
    vec3 sky = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);

    // Check for distant lights (act like sun in sky)
    for (int i = 0; i < num_lights; i++) {
        if (lights[i].type == 1) {  // Distant light
            vec3 sun_dir = unit_vector(lights[i].direction);
            float sun_dot = dot(unit_direction, sun_dir);
            float sun_intensity = powf(fmaxf(0.0f, sun_dot), 512.0f);  // Very sharp sun disk
            sky = sky + sun_intensity * lights[i].color * 2.0f;  // Fixed sun brightness
        }
    }

    return sky;
}

// Debug flags - print light data once from GPU during rendering
__device__ int g_debug_light_printed = 0;
__device__ int g_color_debug_printed = 0;

// Kernel to reset debug flags (call when loading new scene)
__global__ void reset_debug_flags() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        g_debug_light_printed = 0;
        g_color_debug_printed = 0;
        printf("[GPU DEBUG] Reset debug flags\n");
    }
}

// Sample direct lighting with shadow ray
__device__ vec3 sample_direct_light(const hit_record& rec, hitable** world,
                                     LightData* lights, int num_lights,
                                     curandState* local_rand_state) {
    vec3 direct = vec3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_lights; i++) {
        vec3 light_dir;
        float light_dist;
        vec3 light_intensity;

        // DEBUG: Print light data once during rendering
        if (g_debug_light_printed == 0) {
            int old = atomicCAS(&g_debug_light_printed, 0, 1);
            if (old == 0) {
                printf("[RENDER DEBUG] Light %d accessed during rendering:\n", i);
                printf("  type=%d\n", lights[i].type);
                printf("  pos=(%f,%f,%f)\n", lights[i].position.x(), lights[i].position.y(), lights[i].position.z());
                printf("  dir=(%f,%f,%f)\n", lights[i].direction.x(), lights[i].direction.y(), lights[i].direction.z());
                printf("  color=(%f,%f,%f)\n", lights[i].color.r(), lights[i].color.g(), lights[i].color.b());
                printf("  intensity=%f radius=%f cone=%f falloff=%f\n",
                       lights[i].intensity, lights[i].radius, lights[i].cone_angle, lights[i].falloff);
            }
        }

        if (lights[i].type == 1) {
            // Distant light (directional)
            light_dir = lights[i].direction;
            light_dist = 1e30f;  // Infinite distance
            light_intensity = lights[i].color * lights[i].intensity * 0.01f;
        } else {
            // Point light, Sphere light, or any other type - treat as point light
            vec3 to_light = lights[i].position - rec.p;
            light_dist = to_light.length();
            light_dir = to_light / light_dist;
            float falloff = 1.0f / (light_dist * light_dist + 0.1f);
            light_intensity = lights[i].color * (lights[i].intensity * 0.1f) * falloff;
        }

        // Check if surface faces light
        float n_dot_l = dot(rec.normal, light_dir);
        if (n_dot_l <= 0.0f) continue;

        // Shadow ray
        ray shadow_ray(rec.p + rec.normal * 0.001f, light_dir);
        hit_record shadow_rec;
        bool in_shadow = (*world)->hit(shadow_ray, 0.001f, light_dist - 0.001f, shadow_rec);

        if (!in_shadow) {
            direct = direct + light_intensity * n_dot_l;
        }
    }

    return direct;
}

__device__ vec3 color(const ray& r, hitable **world, LightData* lights, int num_lights, curandState *local_rand_state, int max_depth, bool disable_sky) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    vec3 accumulated_light = vec3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < max_depth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            // DEBUG: Print once when we hit something
            if (g_color_debug_printed == 0 && i == 0) {
                int old = atomicCAS(&g_color_debug_printed, 0, 1);
                if (old == 0) {
                    printf("[COLOR DEBUG] First hit! num_lights=%d\n", num_lights);
                    printf("  hit point=(%f,%f,%f)\n", rec.p.x(), rec.p.y(), rec.p.z());
                    printf("  normal=(%f,%f,%f)\n", rec.normal.x(), rec.normal.y(), rec.normal.z());
                }
            }

            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                // Sample direct lighting on all bounces for proper color bleeding
                // The cur_attenuation naturally reduces contribution of later bounces
                if (num_lights > 0) {
                    vec3 direct = sample_direct_light(rec, world, lights, num_lights, local_rand_state);
                    accumulated_light = accumulated_light + cur_attenuation * attenuation * direct;
                }

                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return accumulated_light;
            }
        } else {
            // Hit sky (or black if disabled)
            vec3 sky_color = get_sky_color(cur_ray.direction(), lights, num_lights, disable_sky);
            return accumulated_light + cur_attenuation * sky_color;
        }
    }
    return accumulated_light;
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render_kernel(vec3 *fb, int max_x, int max_y, int ns,
                              camera **cam, hitable **world,
                              LightData* lights, int num_lights,
                              curandState *rand_state, int max_depth, bool disable_sky) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, lights, num_lights, &local_rand_state, max_depth, disable_sky);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    // Clamp before gamma to prevent washed out colors from overexposure
    col[0] = fminf(col[0], 1.0f);
    col[1] = fminf(col[1], 1.0f);
    col[2] = fminf(col[2], 1.0f);
    // Gamma correction
    col[0] = sqrtf(col[0]);
    col[1] = sqrtf(col[1]);
    col[2] = sqrtf(col[2]);
    fb[pixel_index] = col;
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

// Kernel to create spheres, triangles and BVH from CPU-generated data
__global__ void create_world_bvh_kernel(
    hitable **d_list, hitable **d_world, camera **d_camera,
    SphereData *d_sphere_data, int num_spheres,
    TriangleData *d_triangle_data, int num_triangles,
    BVHFlatNode *d_bvh_nodes, int num_nodes,
    int nx, int ny, bool use_bvh)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int prim_idx = 0;

        // Create spheres from CPU-generated data
        for (int i = 0; i < num_spheres; i++) {
            SphereData& sd = d_sphere_data[i];
            material* mat;
            if (sd.material_type == 0) {
                mat = new lambertian(sd.albedo);
            } else if (sd.material_type == 1) {
                mat = new metal(sd.albedo, sd.fuzz_or_ior);
            } else {
                mat = new dielectric(sd.fuzz_or_ior);
            }
            d_list[prim_idx++] = new sphere(sd.center, sd.radius, mat);
        }

        // Create triangles from CPU-generated data
        for (int i = 0; i < num_triangles; i++) {
            TriangleData& td = d_triangle_data[i];
            material* mat;
            if (td.material_type == 0) {
                mat = new lambertian(td.albedo);
            } else if (td.material_type == 1) {
                mat = new metal(td.albedo, td.fuzz_or_ior);
            } else {
                mat = new dielectric(td.fuzz_or_ior);
            }
            if (td.use_vertex_normals) {
                d_list[prim_idx++] = new triangle(td.v0, td.v1, td.v2, td.n0, td.n1, td.n2, mat);
            } else {
                d_list[prim_idx++] = new triangle(td.v0, td.v1, td.v2, mat);
            }
        }

        int total_prims = num_spheres + num_triangles;

        // Create world - BVH or hitable_list
        if (use_bvh) {
            *d_world = new bvh(d_bvh_nodes, d_list, num_nodes, 0);
        } else {
            *d_world = new hitable_list(d_list, total_prims);
        }

        // Create camera
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0f;
        float aperture = 0.1f;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 30.0f,
                               float(nx)/float(ny), aperture, dist_to_focus);
    }
}

__global__ void free_world_bvh_kernel(hitable **d_list, hitable **d_world, camera **d_camera, int num_spheres, int num_triangles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Free spheres (indices 0 to num_spheres-1)
        for (int i = 0; i < num_spheres; i++) {
            delete ((sphere *)d_list[i])->mat_ptr;
            delete d_list[i];
        }
        // Free triangles (indices num_spheres to num_spheres+num_triangles-1)
        for (int i = 0; i < num_triangles; i++) {
            delete ((triangle *)d_list[num_spheres + i])->mat_ptr;
            delete d_list[num_spheres + i];
        }
        delete *d_world;  // Deletes the BVH object
        delete *d_camera;
    }
}

// Kernel to update camera parameters
__global__ void update_camera_kernel(camera **d_camera,
    vec3 lookfrom, vec3 lookat, vec3 vup,
    float vfov, float aspect, float aperture, float focus_dist)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Delete old camera
        if (*d_camera) {
            delete *d_camera;
        }
        // Create new camera with updated parameters
        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist);
    }
}

// Debug kernel to verify light data on GPU
__global__ void debug_lights_kernel(LightData* lights, int num_lights) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU DEBUG] Number of lights: %d\n", num_lights);
        for (int i = 0; i < num_lights; i++) {
            printf("[GPU DEBUG] Light %d:\n", i);
            printf("  type=%d\n", lights[i].type);
            printf("  position=(%f, %f, %f)\n", lights[i].position.x(), lights[i].position.y(), lights[i].position.z());
            printf("  direction=(%f, %f, %f)\n", lights[i].direction.x(), lights[i].direction.y(), lights[i].direction.z());
            printf("  color=(%f, %f, %f)\n", lights[i].color.r(), lights[i].color.g(), lights[i].color.b());
            printf("  intensity=%f\n", lights[i].intensity);
            printf("  radius=%f\n", lights[i].radius);
            printf("  cone_angle=%f\n", lights[i].cone_angle);
            printf("  falloff=%f\n", lights[i].falloff);
        }
    }
}

// Debug kernel to verify BVH data on GPU
__global__ void debug_bvh_kernel(BVHFlatNode* nodes, int num_nodes, hitable** primitives, int num_prims) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU DEBUG] BVH nodes ptr: %p, num_nodes: %d\n", nodes, num_nodes);
        printf("[GPU DEBUG] Primitives ptr: %p, num_prims: %d\n", primitives, num_prims);

        if (nodes == nullptr) {
            printf("[GPU DEBUG] ERROR: nodes is null!\n");
            return;
        }

        // Print first 5 nodes
        for (int i = 0; i < 5 && i < num_nodes; i++) {
            printf("[GPU DEBUG] Node %d: is_leaf=%d, left=%d, right=%d\n",
                   i, nodes[i].is_leaf, nodes[i].left, nodes[i].right);
            printf("[GPU DEBUG]   bounds min: (%f, %f, %f)\n",
                   nodes[i].bounds.minimum.x(), nodes[i].bounds.minimum.y(), nodes[i].bounds.minimum.z());
            printf("[GPU DEBUG]   bounds max: (%f, %f, %f)\n",
                   nodes[i].bounds.maximum.x(), nodes[i].bounds.maximum.y(), nodes[i].bounds.maximum.z());
        }

        // Test accessing a primitive
        if (primitives != nullptr && num_prims > 0) {
            printf("[GPU DEBUG] Testing primitive access...\n");
            hitable* p = primitives[0];
            printf("[GPU DEBUG] primitives[0] = %p\n", p);
        }

        printf("[GPU DEBUG] Debug complete\n");
    }
}

// Simple test kernel - just trace one ray without BVH
__global__ void test_single_ray_kernel(hitable** world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[GPU TEST] Testing single ray...\n");
        ray r(vec3(0, 0, 0), vec3(0, 0, -1));
        hit_record rec;
        printf("[GPU TEST] Calling world->hit()...\n");
        bool hit = (*world)->hit(r, 0.001f, 1e30f, rec);
        printf("[GPU TEST] Hit result: %d\n", hit);
        if (hit) {
            printf("[GPU TEST] Hit at t=%f, point=(%f,%f,%f)\n", rec.t, rec.p.x(), rec.p.y(), rec.p.z());
        }
        printf("[GPU TEST] Test complete\n");
    }
}

// ============= CudaRenderer Implementation =============

CudaRenderer::CudaRenderer()
    : d_framebuffer(nullptr), d_rand_state(nullptr), d_rand_state2(nullptr),
      d_list(nullptr), d_world(nullptr), d_camera(nullptr), d_bvh_nodes(nullptr),
      d_lights(nullptr), num_lights(0),
      width(0), height(0), num_hitables(22*22+1+3), num_bvh_nodes(0),
      num_spheres_in_scene(0), num_triangles_in_scene(0), initialized(false),
      tx(8), ty(8), maxDepth(10), disableSky(false) {}

  CudaRenderer::~CudaRenderer() {
      cleanup();
  }

  bool CudaRenderer::initialize(int w, int h) {
      width = w;
      height = h;

      allocateFramebuffer();

      // Allocate random state
      int num_pixels = width * height;
      checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));
      checkCudaErrors(cudaMalloc(&d_rand_state2, sizeof(curandState)));

      // Initialize random state for world creation
      rand_init<<<1, 1>>>(d_rand_state2);
      checkCudaErrors(cudaDeviceSynchronize());

      // Create world
      createWorld();

      // Initialize per-pixel random states
      dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
      dim3 threads(tx, ty);
      render_init<<<blocks, threads>>>(width, height, d_rand_state);
      checkCudaErrors(cudaDeviceSynchronize());

      initialized = true;
      return true;
  }

  void CudaRenderer::resize(int w, int h) {
      if (w == width && h == height) return;

      freeFramebuffer();
      if (d_rand_state) { cudaFree(d_rand_state); d_rand_state = nullptr; }

      width = w;
      height = h;

      allocateFramebuffer();

      int num_pixels = width * height;
      checkCudaErrors(cudaMalloc(&d_rand_state, num_pixels * sizeof(curandState)));

      dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
      dim3 threads(tx, ty);
      render_init<<<blocks, threads>>>(width, height, d_rand_state);
      checkCudaErrors(cudaDeviceSynchronize());
  }

  // Global debug counter (reset when loading new scene)
  int render_debug_count = 0;

  void CudaRenderer::render(int samplesPerPixel) {
      if (!initialized) return;

      // Debug: Print num_lights on first few renders after each scene load
      if (render_debug_count < 3) {
          std::cout << "[CudaRenderer::render] num_lights=" << num_lights << " d_lights=" << d_lights << std::endl;
          render_debug_count++;
      }

      dim3 blocks((width + tx - 1) / tx, (height + ty - 1) / ty);
      dim3 threads(tx, ty);

      render_kernel<<<blocks, threads>>>(d_framebuffer, width, height, samplesPerPixel,
                                         (camera**)d_camera, (hitable**)d_world,
                                         d_lights, num_lights, d_rand_state, maxDepth, disableSky);
      checkCudaErrors(cudaDeviceSynchronize());
  }

  void CudaRenderer::cleanup() {
      if (!initialized) return;

      freeWorld();
      freeFramebuffer();

      if (d_rand_state) { cudaFree(d_rand_state); d_rand_state = nullptr; }
      if (d_rand_state2) { cudaFree(d_rand_state2); d_rand_state2 = nullptr; }
      if (d_lights) { cudaFree(d_lights); d_lights = nullptr; }

      initialized = false;
  }

  void CudaRenderer::updateCamera(vec3 lookfrom, vec3 lookat, vec3 vup,
                                   float vfov, float aperture, float focus_dist) {
      if (!initialized || d_camera == nullptr) return;

      float aspect = float(width) / float(height);
      update_camera_kernel<<<1, 1>>>(
          (camera**)d_camera,
          lookfrom, lookat, vup,
          vfov, aspect, aperture, focus_dist
      );
      checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaDeviceSynchronize());
  }

  void CudaRenderer::allocateFramebuffer() {
      size_t fb_size = width * height * sizeof(vec3);
      checkCudaErrors(cudaMallocManaged(&d_framebuffer, fb_size));
  }

  void CudaRenderer::freeFramebuffer() {
      if (d_framebuffer) {
          cudaFree(d_framebuffer);
          d_framebuffer = nullptr;
      }
  }

void CudaRenderer::createWorld() {
    std::cout << "[BVH] Generating sphere data on CPU..." << std::endl;
    generateSphereData();
    std::cout << "[BVH] Generated " << h_spheres.size() << " spheres" << std::endl;

    // Handle empty scene - wait for USDA to load
    int total_prims = (int)(h_spheres.size() + h_triangles.size());
    if (total_prims == 0) {
        std::cout << "[BVH] Empty scene - skipping BVH build (waiting for USDA load)" << std::endl;
        num_hitables = 0;
        num_bvh_nodes = 0;
        num_spheres_in_scene = 0;
        num_triangles_in_scene = 0;
        return;
    }

    // Update num_hitables to actual primitive count
    num_hitables = total_prims;
    num_spheres_in_scene = (int)h_spheres.size();
    num_triangles_in_scene = (int)h_triangles.size();

    // Build BVH on CPU
    std::cout << "[BVH] Building BVH on CPU..." << std::endl;
    h_bvh_nodes.clear();
    h_bvh_nodes.reserve(2 * num_hitables);  // Pre-allocate to prevent reallocations
    std::vector<int> prim_indices(num_hitables);
    for (int i = 0; i < num_hitables; i++) {
        prim_indices[i] = i;
    }
    buildBVH(prim_indices, 0, num_hitables);
    num_bvh_nodes = (int)h_bvh_nodes.size();
    std::cout << "[BVH] Built " << num_bvh_nodes << " BVH nodes" << std::endl;

    // Validate BVH structure
    int leaf_count = 0;
    int internal_count = 0;
    int invalid_count = 0;
    for (int i = 0; i < num_bvh_nodes; i++) {
        const BVHBuildNode& n = h_bvh_nodes[i];
        if (n.is_leaf != 0) {  // is_leaf is now int (1=true, 0=false)
            leaf_count++;
            if (n.left < 0 || n.left >= num_hitables) {
                std::cerr << "[BVH] ERROR: Leaf node " << i << " has invalid prim_idx " << n.left << std::endl;
                invalid_count++;
            }
        } else {
            internal_count++;
            if (n.left < 0 || n.left >= num_bvh_nodes) {
                std::cerr << "[BVH] ERROR: Internal node " << i << " has invalid left " << n.left << std::endl;
                invalid_count++;
            }
            if (n.right < 0 || n.right >= num_bvh_nodes) {
                std::cerr << "[BVH] ERROR: Internal node " << i << " has invalid right " << n.right << std::endl;
                invalid_count++;
            }
        }
    }
    std::cout << "[BVH] Validation: " << leaf_count << " leaves, " << internal_count << " internal, " << invalid_count << " invalid" << std::endl;

    // Copy BVH and sphere data to device
    copyBVHToDevice();
    std::cout << "[BVH] Copied to GPU, creating world..." << std::endl;

    // Allocate device arrays
    checkCudaErrors(cudaMalloc(&d_list, num_hitables * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera*)));

    // Copy sphere data to device
    int num_spheres = (int)h_spheres.size();
    int num_triangles = (int)h_triangles.size();

    SphereData* d_sphere_data = nullptr;
    if (num_spheres > 0) {
        checkCudaErrors(cudaMalloc(&d_sphere_data, num_spheres * sizeof(SphereData)));
        checkCudaErrors(cudaMemcpy(d_sphere_data, h_spheres.data(),
                                   num_spheres * sizeof(SphereData), cudaMemcpyHostToDevice));
    }

    TriangleData* d_triangle_data = nullptr;
    if (num_triangles > 0) {
        checkCudaErrors(cudaMalloc(&d_triangle_data, num_triangles * sizeof(TriangleData)));
        checkCudaErrors(cudaMemcpy(d_triangle_data, h_triangles.data(),
                                   num_triangles * sizeof(TriangleData), cudaMemcpyHostToDevice));
    }

    // Create world with BVH acceleration
    bool use_bvh = true;
    create_world_bvh_kernel<<<1, 1>>>(
        (hitable**)d_list, (hitable**)d_world, (camera**)d_camera,
        d_sphere_data, num_spheres,
        d_triangle_data, num_triangles,
        d_bvh_nodes, num_bvh_nodes,
        width, height, use_bvh);
    checkCudaErrors(cudaDeviceSynchronize());

    // Free temporary data (primitives are now created on GPU)
    if (d_sphere_data) cudaFree(d_sphere_data);
    if (d_triangle_data) cudaFree(d_triangle_data);
    std::cout << "[BVH] World creation complete with " << num_bvh_nodes << " BVH nodes" << std::endl;
}

void CudaRenderer::freeWorld() {
    if (d_list) {
        free_world_bvh_kernel<<<1, 1>>>((hitable**)d_list, (hitable**)d_world,
                                         (camera**)d_camera, num_spheres_in_scene, num_triangles_in_scene);
        cudaDeviceSynchronize();
        cudaFree(d_list); d_list = nullptr;
        cudaFree(d_world); d_world = nullptr;
        cudaFree(d_camera); d_camera = nullptr;
    }
    if (d_bvh_nodes) {
        cudaFree(d_bvh_nodes);
        d_bvh_nodes = nullptr;
    }
    num_spheres_in_scene = 0;
    num_triangles_in_scene = 0;
    h_spheres.clear();
    h_triangles.clear();
    h_meshes.clear();
    h_bvh_nodes.clear();
}

// Generate sphere data on CPU - now starts empty (scenes loaded via USDA)
void CudaRenderer::generateSphereData() {
    // Start with empty scene - USDA loader will populate it
    h_spheres.clear();
    h_triangles.clear();
    h_meshes.clear();
    std::cout << "[CudaRenderer] Starting with empty scene (waiting for USDA load)" << std::endl;
}

// Helper: Get bounding box for a primitive (sphere or triangle)
// Indices 0 to num_spheres-1 are spheres, num_spheres to end are triangles
aabb CudaRenderer::getPrimitiveBounds(int prim_idx) const {
    int num_spheres = (int)h_spheres.size();
    if (prim_idx < num_spheres) {
        // It's a sphere
        const SphereData& s = h_spheres[prim_idx];
        return aabb(s.center - vec3(s.radius, s.radius, s.radius),
                    s.center + vec3(s.radius, s.radius, s.radius));
    } else {
        // It's a triangle
        int tri_idx = prim_idx - num_spheres;
        const TriangleData& t = h_triangles[tri_idx];
        const float pad = 0.0001f;
        vec3 min_pt(
            fminf(fminf(t.v0.x(), t.v1.x()), t.v2.x()) - pad,
            fminf(fminf(t.v0.y(), t.v1.y()), t.v2.y()) - pad,
            fminf(fminf(t.v0.z(), t.v1.z()), t.v2.z()) - pad
        );
        vec3 max_pt(
            fmaxf(fmaxf(t.v0.x(), t.v1.x()), t.v2.x()) + pad,
            fmaxf(fmaxf(t.v0.y(), t.v1.y()), t.v2.y()) + pad,
            fmaxf(fmaxf(t.v0.z(), t.v1.z()), t.v2.z()) + pad
        );
        return aabb(min_pt, max_pt);
    }
}

// Helper: Get centroid for a primitive (used for BVH sorting)
vec3 CudaRenderer::getPrimitiveCentroid(int prim_idx) const {
    int num_spheres = (int)h_spheres.size();
    if (prim_idx < num_spheres) {
        return h_spheres[prim_idx].center;
    } else {
        int tri_idx = prim_idx - num_spheres;
        const TriangleData& t = h_triangles[tri_idx];
        return (t.v0 + t.v1 + t.v2) / 3.0f;
    }
}

// Recursive BVH build (returns node index)
// Handles both spheres (indices 0 to num_spheres-1) and triangles (num_spheres onwards)
int CudaRenderer::buildBVH(std::vector<int>& prim_indices, int start, int end) {
    int node_idx = (int)h_bvh_nodes.size();
    h_bvh_nodes.push_back(BVHBuildNode());
    // NOTE: Do NOT keep a reference to h_bvh_nodes[node_idx] - vector may reallocate!

    // Compute bounding box for this range
    aabb bounds;
    bool first = true;
    for (int i = start; i < end; i++) {
        int prim = prim_indices[i];
        aabb prim_box = getPrimitiveBounds(prim);
        if (first) {
            bounds = prim_box;
            first = false;
        } else {
            bounds = surrounding_box(bounds, prim_box);
        }
    }

    int count = end - start;
    if (count == 1) {
        // Leaf node - store directly, no recursion
        h_bvh_nodes[node_idx].bounds = bounds;
        h_bvh_nodes[node_idx].is_leaf = 1;  // Use int: 1 = true
        h_bvh_nodes[node_idx].prim_idx = prim_indices[start];
        h_bvh_nodes[node_idx].left = prim_indices[start];  // Store prim index in left for GPU
        h_bvh_nodes[node_idx].right = -1;
    } else {
        // Find longest axis and sort
        vec3 extent = bounds.maximum - bounds.minimum;
        int axis = 0;
        if (extent.y() > extent.x()) axis = 1;
        if (extent.z() > extent[axis]) axis = 2;

        // Sort primitives along axis using centroid
        std::sort(prim_indices.begin() + start, prim_indices.begin() + end,
            [this, axis](int a, int b) {
                return getPrimitiveCentroid(a)[axis] < getPrimitiveCentroid(b)[axis];
            });

        // Split in middle
        int mid = start + count / 2;

        // Recursive calls - vector may reallocate, so get indices first
        int left_idx = buildBVH(prim_indices, start, mid);
        int right_idx = buildBVH(prim_indices, mid, end);

        // Now safe to write back to our node
        h_bvh_nodes[node_idx].bounds = bounds;
        h_bvh_nodes[node_idx].is_leaf = 0;  // Use int: 0 = false
        h_bvh_nodes[node_idx].prim_idx = -1;
        h_bvh_nodes[node_idx].left = left_idx;
        h_bvh_nodes[node_idx].right = right_idx;
    }

    return node_idx;
}

// Copy BVH nodes to device
void CudaRenderer::copyBVHToDevice() {
    // Convert BVHBuildNode to BVHFlatNode
    std::vector<BVHFlatNode> flat_nodes(h_bvh_nodes.size());
    for (size_t i = 0; i < h_bvh_nodes.size(); i++) {
        flat_nodes[i].bounds = h_bvh_nodes[i].bounds;
        flat_nodes[i].left = h_bvh_nodes[i].left;
        flat_nodes[i].right = h_bvh_nodes[i].right;
        flat_nodes[i].is_leaf = h_bvh_nodes[i].is_leaf;
    }

    // Allocate and copy to device
    checkCudaErrors(cudaMalloc(&d_bvh_nodes, flat_nodes.size() * sizeof(BVHFlatNode)));
    checkCudaErrors(cudaMemcpy(d_bvh_nodes, flat_nodes.data(),
                               flat_nodes.size() * sizeof(BVHFlatNode), cudaMemcpyHostToDevice));
}

//=============================================================================
// USDA EXPORT
//=============================================================================

bool CudaRenderer::exportSceneToUSDA(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[CudaRenderer] Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    std::cout << "[CudaRenderer] Exporting " << h_spheres.size() << " spheres to " << filename << std::endl;

    // Header
    file << "#usda 1.0\n";
    file << "(\n";
    file << "    defaultPrim = \"World\"\n";
    file << "    upAxis = \"Y\"\n";
    file << ")\n\n";

    file << "def Xform \"World\"\n{\n";

    // Collect unique materials
    std::map<std::string, int> materialMap;  // name -> index in h_spheres

    for (size_t i = 0; i < h_spheres.size(); i++) {
        const SphereData& s = h_spheres[i];
        std::string matName;

        if (s.material_type == 0) {
            // Lambertian - name by color
            char buf[64];
            snprintf(buf, sizeof(buf), "Lambertian_%d_%d_%d",
                     int(s.albedo.r() * 255), int(s.albedo.g() * 255), int(s.albedo.b() * 255));
            matName = buf;
        } else if (s.material_type == 1) {
            // Metal
            char buf[64];
            snprintf(buf, sizeof(buf), "Metal_%d_%d_%d_f%d",
                     int(s.albedo.r() * 255), int(s.albedo.g() * 255), int(s.albedo.b() * 255),
                     int(s.fuzz_or_ior * 100));
            matName = buf;
        } else {
            // Dielectric
            char buf[64];
            snprintf(buf, sizeof(buf), "Glass_ior%d", int(s.fuzz_or_ior * 100));
            matName = buf;
        }

        if (materialMap.find(matName) == materialMap.end()) {
            materialMap[matName] = (int)i;
        }
    }

    // Write spheres
    for (size_t i = 0; i < h_spheres.size(); i++) {
        const SphereData& s = h_spheres[i];

        // Generate material name
        std::string matName;
        if (s.material_type == 0) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Lambertian_%d_%d_%d",
                     int(s.albedo.r() * 255), int(s.albedo.g() * 255), int(s.albedo.b() * 255));
            matName = buf;
        } else if (s.material_type == 1) {
            char buf[64];
            snprintf(buf, sizeof(buf), "Metal_%d_%d_%d_f%d",
                     int(s.albedo.r() * 255), int(s.albedo.g() * 255), int(s.albedo.b() * 255),
                     int(s.fuzz_or_ior * 100));
            matName = buf;
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "Glass_ior%d", int(s.fuzz_or_ior * 100));
            matName = buf;
        }

        file << "    def Xform \"Sphere_" << i << "\"\n";
        file << "    {\n";
        file << "        float3 xformOp:translate = (" << s.center.x() << ", " << s.center.y() << ", " << s.center.z() << ")\n";
        file << "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n\n";
        file << "        def Sphere \"Geom\"\n";
        file << "        {\n";
        file << "            double radius = " << s.radius << "\n";
        file << "            rel material:binding = </World/Materials/" << matName << ">\n";
        file << "        }\n";
        file << "    }\n\n";
    }

    // Write light
    file << "    def SphereLight \"MainLight\"\n";
    file << "    {\n";
    file << "        float3 xformOp:translate = (0, 20, 0)\n";
    file << "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
    file << "        float inputs:intensity = 1000\n";
    file << "        color3f inputs:color = (1, 1, 1)\n";
    file << "    }\n\n";

    // Write materials
    file << "    def Scope \"Materials\"\n";
    file << "    {\n";

    for (const auto& pair : materialMap) {
        const std::string& matName = pair.first;
        const SphereData& s = h_spheres[pair.second];

        file << "        def Material \"" << matName << "\"\n";
        file << "        {\n";
        file << "            def Shader \"PBRShader\"\n";
        file << "            {\n";
        file << "                uniform token info:id = \"UsdPreviewSurface\"\n";
        file << "                color3f inputs:diffuseColor = (" << s.albedo.r() << ", " << s.albedo.g() << ", " << s.albedo.b() << ")\n";

        if (s.material_type == 0) {
            // Lambertian - diffuse
            file << "                float inputs:roughness = 1.0\n";
            file << "                float inputs:metallic = 0.0\n";
        } else if (s.material_type == 1) {
            // Metal
            file << "                float inputs:roughness = " << s.fuzz_or_ior << "\n";
            file << "                float inputs:metallic = 1.0\n";
        } else {
            // Dielectric (glass)
            file << "                float inputs:roughness = 0.0\n";
            file << "                float inputs:metallic = 0.0\n";
            file << "                float inputs:ior = " << s.fuzz_or_ior << "\n";
            file << "                float inputs:opacity = 0.1\n";
        }

        file << "            }\n";
        file << "        }\n\n";
    }

    file << "    }\n";  // Close Materials
    file << "}\n";      // Close World

    file.close();
    std::cout << "[CudaRenderer] Exported scene with " << materialMap.size() << " unique materials" << std::endl;
    return true;
}

//=============================================================================
// USDA LOADING
//=============================================================================

#include "usd/USDAParser.h"

bool CudaRenderer::loadSceneFromUSDA(const std::string& filename) {
    usd::USDAParser parser;
    usd::USDAScene scene;

    if (!parser.parse(filename, scene)) {
        std::cerr << "[CudaRenderer] Failed to parse USDA: " << parser.getError() << std::endl;
        return false;
    }

    // Clear existing scene
    freeWorld();
    h_spheres.clear();
    h_triangles.clear();
    h_bvh_nodes.clear();

    // Apply scene settings
    disableSky = scene.disableSky;
    std::cout << "[CudaRenderer] Scene disableSky = " << (disableSky ? "true" : "false") << std::endl;

    // Build material lookup map
    std::map<std::string, usd::USDAMaterial> materialLookup;
    for (const auto& mat : scene.materials) {
        materialLookup[mat.name] = mat;
        std::cout << "[CudaRenderer] Material: '" << mat.name << "' color=("
                  << mat.diffuseColor.r << "," << mat.diffuseColor.g << "," << mat.diffuseColor.b
                  << ") metallic=" << mat.metallic << std::endl;
    }

    std::cout << "[CudaRenderer] Loading " << scene.spheres.size() << " spheres from USDA" << std::endl;

    // Convert spheres
    int sphereIdx = 0;
    int foundMats = 0, missingMats = 0;
    for (const auto& usdSphere : scene.spheres) {
        SphereData s;
        s.center = vec3(usdSphere.center.x, usdSphere.center.y, usdSphere.center.z);
        s.radius = usdSphere.radius;
        s.name = usdSphere.name;

        // Debug: print first few sphere material bindings
        if (sphereIdx < 5) {
            std::cout << "[CudaRenderer] Sphere " << sphereIdx << " materialBinding='"
                      << usdSphere.materialBinding << "'" << std::endl;
        }
        sphereIdx++;

        // Find material
        auto it = materialLookup.find(usdSphere.materialBinding);
        if (it != materialLookup.end()) {
            foundMats++;
            const usd::USDAMaterial& mat = it->second;

            if (mat.opacity < 0.5f) {
                // Dielectric (glass) - detected by low opacity
                s.material_type = 2;
                s.albedo = vec3(1.0f, 1.0f, 1.0f);
                s.fuzz_or_ior = (mat.ior > 1.0f && mat.ior != 1.5f) ? mat.ior : 1.5f;
                if (sphereIdx <= 5) std::cout << "  -> Glass (opacity=" << mat.opacity << ")" << std::endl;
            } else if (mat.metallic > 0.5f) {
                // Metal
                s.material_type = 1;
                s.albedo = vec3(mat.diffuseColor.r, mat.diffuseColor.g, mat.diffuseColor.b);
                s.fuzz_or_ior = mat.roughness;
                if (sphereIdx <= 5) std::cout << "  -> Metal (metallic=" << mat.metallic << ")" << std::endl;
            } else {
                // Lambertian
                s.material_type = 0;
                s.albedo = vec3(mat.diffuseColor.r, mat.diffuseColor.g, mat.diffuseColor.b);
                s.fuzz_or_ior = 0.0f;
                if (sphereIdx <= 5) std::cout << "  -> Lambertian color=(" << s.albedo.r() << "," << s.albedo.g() << "," << s.albedo.b() << ")" << std::endl;
            }
        } else {
            // Default: gray lambertian
            missingMats++;
            if (missingMats <= 5) {
                std::cout << "[CudaRenderer] WARNING: No material found for binding '"
                          << usdSphere.materialBinding << "'" << std::endl;
            }
            s.material_type = 0;
            s.albedo = vec3(0.5f, 0.5f, 0.5f);
            s.fuzz_or_ior = 0.0f;
        }

        h_spheres.push_back(s);
    }

    std::cout << "[CudaRenderer] Material lookup: " << foundMats << " found, "
              << missingMats << " missing" << std::endl;

    // Convert meshes to triangles
    std::cout << "[CudaRenderer] Loading " << scene.meshes.size() << " meshes from USDA" << std::endl;
    h_meshes.clear();

    for (const auto& mesh : scene.meshes) {
        // Track mesh metadata for material controls
        MeshData meshData;
        meshData.name = mesh.name;
        meshData.triangleStart = static_cast<int>(h_triangles.size());

        // Look up material for this mesh
        int mat_type = 0;
        vec3 albedo(0.5f, 0.5f, 0.5f);
        float fuzz_or_ior = 0.0f;

        auto matIt = materialLookup.find(mesh.materialBinding);
        if (matIt != materialLookup.end()) {
            const usd::USDAMaterial& mat = matIt->second;
            if (mat.opacity < 0.5f) {
                mat_type = 2;  // Dielectric
                albedo = vec3(1.0f, 1.0f, 1.0f);
                fuzz_or_ior = (mat.ior > 1.0f) ? mat.ior : 1.5f;
            } else if (mat.metallic > 0.5f) {
                mat_type = 1;  // Metal
                albedo = vec3(mat.diffuseColor.r, mat.diffuseColor.g, mat.diffuseColor.b);
                fuzz_or_ior = mat.roughness;
            } else {
                mat_type = 0;  // Lambertian
                albedo = vec3(mat.diffuseColor.r, mat.diffuseColor.g, mat.diffuseColor.b);
                fuzz_or_ior = 0.0f;
            }
        } else if (!mesh.materialBinding.empty()) {
            std::cout << "[CudaRenderer] WARNING: No material '" << mesh.materialBinding
                      << "' for mesh '" << mesh.name << "'" << std::endl;
        }

        // Store material in mesh metadata
        meshData.material_type = mat_type;
        meshData.albedo = albedo;
        meshData.fuzz_or_ior = fuzz_or_ior;

        // Transform vertices (apply mesh transform)
        std::vector<vec3> transformedPoints;
        transformedPoints.reserve(mesh.points.size());
        for (const auto& p : mesh.points) {
            vec3 tp(
                p.x * mesh.scale.x + mesh.translate.x,
                p.y * mesh.scale.y + mesh.translate.y,
                p.z * mesh.scale.z + mesh.translate.z
            );
            transformedPoints.push_back(tp);
        }

        // Transform normals if present
        std::vector<vec3> transformedNormals;
        if (!mesh.normals.empty()) {
            transformedNormals.reserve(mesh.normals.size());
            for (const auto& n : mesh.normals) {
                // Normals don't get translated, only rotated/scaled
                // For uniform scale, just use the normal directly
                vec3 tn(n.x, n.y, n.z);
                transformedNormals.push_back(unit_vector(tn));
            }
        }

        // Triangulate faces
        int indexOffset = 0;
        for (size_t faceIdx = 0; faceIdx < mesh.faceVertexCounts.size(); faceIdx++) {
            int vertCount = mesh.faceVertexCounts[faceIdx];

            if (vertCount < 3) {
                indexOffset += vertCount;
                continue;
            }

            // Get vertex indices for this face
            std::vector<int> faceIndices;
            for (int v = 0; v < vertCount; v++) {
                faceIndices.push_back(mesh.faceVertexIndices[indexOffset + v]);
            }

            // Triangulate using fan triangulation (works for convex polygons)
            for (int v = 1; v < vertCount - 1; v++) {
                TriangleData tri;
                int i0 = faceIndices[0];
                int i1 = faceIndices[v];
                int i2 = faceIndices[v + 1];

                tri.v0 = transformedPoints[i0];
                tri.v1 = transformedPoints[i1];
                tri.v2 = transformedPoints[i2];

                // Check if we have per-vertex normals
                if (!transformedNormals.empty() && transformedNormals.size() == mesh.points.size()) {
                    tri.n0 = transformedNormals[i0];
                    tri.n1 = transformedNormals[i1];
                    tri.n2 = transformedNormals[i2];
                    tri.use_vertex_normals = true;
                } else {
                    // Compute flat normal
                    vec3 edge1 = tri.v1 - tri.v0;
                    vec3 edge2 = tri.v2 - tri.v0;
                    vec3 normal = unit_vector(cross(edge1, edge2));
                    tri.n0 = tri.n1 = tri.n2 = normal;
                    tri.use_vertex_normals = false;
                }

                tri.material_type = mat_type;
                tri.albedo = albedo;
                tri.fuzz_or_ior = fuzz_or_ior;

                h_triangles.push_back(tri);
            }

            indexOffset += vertCount;
        }

        // Store mesh metadata
        meshData.triangleCount = static_cast<int>(h_triangles.size()) - meshData.triangleStart;
        if (meshData.triangleCount > 0) {
            h_meshes.push_back(meshData);
        }

        std::cout << "[CudaRenderer] Mesh '" << mesh.name << "': " << mesh.faceVertexCounts.size()
                  << " faces -> " << meshData.triangleCount << " triangles (total: " << h_triangles.size() << ")" << std::endl;
    }

    if (h_spheres.empty() && h_triangles.empty()) {
        std::cerr << "[CudaRenderer] No geometry found in USDA file" << std::endl;
        return false;
    }

    int num_spheres = (int)h_spheres.size();
    int num_triangles = (int)h_triangles.size();
    num_hitables = num_spheres + num_triangles;
    num_spheres_in_scene = num_spheres;      // Store for cleanup
    num_triangles_in_scene = num_triangles;  // Store for cleanup

    std::cout << "[CudaRenderer] Total primitives: " << num_spheres << " spheres + "
              << num_triangles << " triangles = " << num_hitables << std::endl;

    // Build BVH
    std::cout << "[CudaRenderer] Building BVH for " << num_hitables << " primitives..." << std::endl;
    h_bvh_nodes.clear();
    h_bvh_nodes.reserve(2 * num_hitables);
    std::vector<int> prim_indices(num_hitables);
    for (int i = 0; i < num_hitables; i++) {
        prim_indices[i] = i;
    }
    buildBVH(prim_indices, 0, num_hitables);
    num_bvh_nodes = (int)h_bvh_nodes.size();

    // Copy to GPU and create world
    copyBVHToDevice();

    checkCudaErrors(cudaMalloc(&d_list, num_hitables * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera*)));

    // Copy sphere data to device
    SphereData* d_sphere_data = nullptr;
    if (num_spheres > 0) {
        checkCudaErrors(cudaMalloc(&d_sphere_data, num_spheres * sizeof(SphereData)));
        checkCudaErrors(cudaMemcpy(d_sphere_data, h_spheres.data(),
                                   num_spheres * sizeof(SphereData), cudaMemcpyHostToDevice));
    }

    // Copy triangle data to device
    TriangleData* d_triangle_data = nullptr;
    if (num_triangles > 0) {
        checkCudaErrors(cudaMalloc(&d_triangle_data, num_triangles * sizeof(TriangleData)));
        checkCudaErrors(cudaMemcpy(d_triangle_data, h_triangles.data(),
                                   num_triangles * sizeof(TriangleData), cudaMemcpyHostToDevice));
    }

    bool use_bvh = true;
    create_world_bvh_kernel<<<1, 1>>>(
        (hitable**)d_list, (hitable**)d_world, (camera**)d_camera,
        d_sphere_data, num_spheres,
        d_triangle_data, num_triangles,
        d_bvh_nodes, num_bvh_nodes,
        width, height, use_bvh);
    checkCudaErrors(cudaDeviceSynchronize());

    // Free temporary data (primitives are now created on GPU)
    if (d_sphere_data) cudaFree(d_sphere_data);
    if (d_triangle_data) cudaFree(d_triangle_data);

    // Load lights from USDA
    h_lights.clear();
    if (d_lights) {
        cudaFree(d_lights);
        d_lights = nullptr;
    }

    std::cout << "[CudaRenderer] sizeof(LightData) = " << sizeof(LightData) << " bytes" << std::endl;
    std::cout << "[CudaRenderer] sizeof(vec3) = " << sizeof(vec3) << " bytes" << std::endl;
    std::cout << "[CudaRenderer] sizeof(int) = " << sizeof(int) << " bytes" << std::endl;
    std::cout << "[CudaRenderer] sizeof(float) = " << sizeof(float) << " bytes" << std::endl;

    for (const auto& usdLight : scene.lights) {
        LightData light = {};  // Zero-initialize all fields
        switch (usdLight.type) {
            case usd::USDALightType::Point:
                light.type = 0;
                break;
            case usd::USDALightType::Distant:
                light.type = 1;
                break;
            case usd::USDALightType::Sphere:
                light.type = 2;
                break;
            case usd::USDALightType::Rect:
                light.type = 3;
                break;
            case usd::USDALightType::Spot:
                light.type = 4;
                break;
        }
        light.position = vec3(usdLight.position.x, usdLight.position.y, usdLight.position.z);
        // For distant lights, use position as direction if set, otherwise default
        if (usdLight.type == usd::USDALightType::Distant) {
            float pos_len = light.position.length();
            if (pos_len > 0.001f) {
                light.direction = light.position / pos_len;  // Position points toward the sun
            } else {
                // Default sun direction: upper-right
                light.direction = unit_vector(vec3(1.0f, 1.0f, 0.5f));
            }
        } else if (usdLight.type == usd::USDALightType::Spot) {
            // For spotlights, direction is where the light is pointing (no negation)
            vec3 dir_raw = vec3(usdLight.direction.x, usdLight.direction.y, usdLight.direction.z);
            float dir_len = dir_raw.length();
            std::cout << "[CudaRenderer] SpotLight raw direction: (" << usdLight.direction.x << ","
                      << usdLight.direction.y << "," << usdLight.direction.z << ") len=" << dir_len << std::endl;
            if (dir_len > 0.001f) {
                light.direction = dir_raw / dir_len;
            } else {
                light.direction = vec3(0, -1, 0);  // Default: pointing down
            }
        } else {
            light.direction = vec3(-usdLight.direction.x, -usdLight.direction.y, -usdLight.direction.z);
        }
        light.color = vec3(usdLight.color.r, usdLight.color.g, usdLight.color.b);
        light.intensity = usdLight.intensity;
        light.radius = (usdLight.radius > 0.01f) ? usdLight.radius : 0.5f;  // Default radius if not set
        // Spotlight parameters
        light.cone_angle = usdLight.coneAngle * 3.14159265f / 180.0f;  // Convert degrees to radians
        light.falloff = usdLight.coneFalloff;
        h_lights.push_back(light);

        std::cout << "[CudaRenderer] Light: type=" << light.type
                  << " pos=(" << light.position.x() << "," << light.position.y() << "," << light.position.z() << ")"
                  << " dir=(" << light.direction.x() << "," << light.direction.y() << "," << light.direction.z() << ")"
                  << " color=(" << light.color.r() << "," << light.color.g() << "," << light.color.b() << ")"
                  << " intensity=" << light.intensity
                  << " radius=" << light.radius
                  << " cone_angle=" << light.cone_angle
                  << " falloff=" << light.falloff
                  << std::endl;
    }

    num_lights = (int)h_lights.size();
    std::cout << "[CudaRenderer] SET num_lights = " << num_lights << " (address: " << &num_lights << ")" << std::endl;
    if (num_lights > 0) {
        checkCudaErrors(cudaMalloc(&d_lights, num_lights * sizeof(LightData)));
        checkCudaErrors(cudaMemcpy(d_lights, h_lights.data(),
                                   num_lights * sizeof(LightData), cudaMemcpyHostToDevice));

        // Debug: Print light data on GPU to verify memory copy
        std::cout << "[CudaRenderer] Running GPU debug kernel for lights..." << std::endl;
        debug_lights_kernel<<<1, 1>>>(d_lights, num_lights);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    std::cout << "[CudaRenderer] After GPU copy, num_lights = " << num_lights << std::endl;
    std::cout << "[CudaRenderer] Loaded scene: " << h_spheres.size() << " spheres, "
              << h_triangles.size() << " triangles, "
              << num_bvh_nodes << " BVH nodes, " << num_lights << " lights" << std::endl;

    // Reset render debug counter so we see debug output after loading new scene
    extern int render_debug_count;
    render_debug_count = 0;

    // Reset GPU debug flags
    reset_debug_flags<<<1, 1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "[CudaRenderer] Reset debug counters, next renders will show full debug" << std::endl;

    return true;
}

void CudaRenderer::setLights(const std::vector<LightData>& lights) {
    h_lights = lights;
    num_lights = static_cast<int>(h_lights.size());

    // Free old light data
    if (d_lights) {
        checkCudaErrors(cudaFree(d_lights));
        d_lights = nullptr;
    }

    // Allocate and copy new light data
    if (num_lights > 0) {
        checkCudaErrors(cudaMalloc(&d_lights, num_lights * sizeof(LightData)));
        checkCudaErrors(cudaMemcpy(d_lights, h_lights.data(), num_lights * sizeof(LightData), cudaMemcpyHostToDevice));
    }
}

void CudaRenderer::setMaterials(const std::vector<MaterialData>& materials) {
    // Update h_spheres with new material data
    std::cout << "[CudaRenderer::setMaterials] Updating " << materials.size() << " materials, h_spheres has " << h_spheres.size() << std::endl;
    for (size_t i = 0; i < materials.size() && i < h_spheres.size(); ++i) {
        h_spheres[i].material_type = materials[i].type;
        h_spheres[i].albedo = materials[i].albedo;
        h_spheres[i].fuzz_or_ior = (materials[i].type == 1) ? materials[i].fuzz : materials[i].ior;
    }

    // Need to rebuild the entire GPU scene to pick up material changes
    // Save CPU-side data since freeWorld() clears it
    auto savedSpheres = std::move(h_spheres);
    auto savedTriangles = std::move(h_triangles);
    auto savedMeshes = std::move(h_meshes);
    auto savedBvhNodes = std::move(h_bvh_nodes);

    // Free GPU world (this will clear h_spheres/h_triangles/h_meshes/h_bvh_nodes)
    freeWorld();

    // Restore CPU-side data with updated materials
    h_spheres = std::move(savedSpheres);
    h_triangles = std::move(savedTriangles);
    h_meshes = std::move(savedMeshes);
    h_bvh_nodes = std::move(savedBvhNodes);

    // Rebuild GPU scene directly (don't call createWorld which calls generateSphereData)
    rebuildGPUWorld();
}

void CudaRenderer::rebuildGPUWorld() {
    // Rebuild GPU world from existing h_spheres/h_triangles/h_bvh_nodes
    // This is like createWorld() but doesn't call generateSphereData()

    int total_prims = (int)(h_spheres.size() + h_triangles.size());
    if (total_prims == 0) {
        std::cout << "[CudaRenderer::rebuildGPUWorld] No primitives to rebuild" << std::endl;
        return;
    }

    num_hitables = total_prims;
    num_spheres_in_scene = (int)h_spheres.size();
    num_triangles_in_scene = (int)h_triangles.size();
    num_bvh_nodes = (int)h_bvh_nodes.size();

    std::cout << "[CudaRenderer::rebuildGPUWorld] Rebuilding with " << num_spheres_in_scene
              << " spheres, " << num_triangles_in_scene << " triangles, " << num_bvh_nodes << " BVH nodes" << std::endl;

    // Copy BVH to device
    copyBVHToDevice();

    // Allocate device arrays
    checkCudaErrors(cudaMalloc(&d_list, num_hitables * sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hitable*)));
    checkCudaErrors(cudaMalloc(&d_camera, sizeof(camera*)));

    // Copy sphere data to device
    SphereData* d_sphere_data = nullptr;
    if (num_spheres_in_scene > 0) {
        checkCudaErrors(cudaMalloc(&d_sphere_data, num_spheres_in_scene * sizeof(SphereData)));
        checkCudaErrors(cudaMemcpy(d_sphere_data, h_spheres.data(),
                                   num_spheres_in_scene * sizeof(SphereData), cudaMemcpyHostToDevice));
    }

    // Copy triangle data to device
    TriangleData* d_triangle_data = nullptr;
    if (num_triangles_in_scene > 0) {
        checkCudaErrors(cudaMalloc(&d_triangle_data, num_triangles_in_scene * sizeof(TriangleData)));
        checkCudaErrors(cudaMemcpy(d_triangle_data, h_triangles.data(),
                                   num_triangles_in_scene * sizeof(TriangleData), cudaMemcpyHostToDevice));
    }

    // Create world with BVH acceleration
    bool use_bvh = true;
    create_world_bvh_kernel<<<1, 1>>>(
        (hitable**)d_list, (hitable**)d_world, (camera**)d_camera,
        d_sphere_data, num_spheres_in_scene,
        d_triangle_data, num_triangles_in_scene,
        d_bvh_nodes, num_bvh_nodes,
        width, height, use_bvh);
    checkCudaErrors(cudaDeviceSynchronize());

    // Free temporary data (primitives are now created on GPU)
    if (d_sphere_data) cudaFree(d_sphere_data);
    if (d_triangle_data) cudaFree(d_triangle_data);

    std::cout << "[CudaRenderer::rebuildGPUWorld] World rebuilt successfully" << std::endl;
}

void CudaRenderer::setMeshMaterial(int meshIndex, int materialType, vec3 albedo, float fuzz_or_ior) {
    if (meshIndex < 0 || meshIndex >= static_cast<int>(h_meshes.size())) {
        std::cerr << "[CudaRenderer::setMeshMaterial] Invalid mesh index: " << meshIndex << std::endl;
        return;
    }

    // Update mesh metadata
    MeshData& mesh = h_meshes[meshIndex];
    mesh.material_type = materialType;
    mesh.albedo = albedo;
    mesh.fuzz_or_ior = fuzz_or_ior;

    // Update all triangles belonging to this mesh
    std::cout << "[CudaRenderer::setMeshMaterial] Updating mesh " << meshIndex << " (" << mesh.name
              << "): triangles " << mesh.triangleStart << " to " << (mesh.triangleStart + mesh.triangleCount - 1) << std::endl;

    for (int i = mesh.triangleStart; i < mesh.triangleStart + mesh.triangleCount; ++i) {
        if (i >= 0 && i < static_cast<int>(h_triangles.size())) {
            h_triangles[i].material_type = materialType;
            h_triangles[i].albedo = albedo;
            h_triangles[i].fuzz_or_ior = fuzz_or_ior;
        }
    }

    // Rebuild GPU world with updated triangles
    auto savedSpheres = std::move(h_spheres);
    auto savedTriangles = std::move(h_triangles);
    auto savedMeshes = std::move(h_meshes);
    auto savedBvhNodes = std::move(h_bvh_nodes);

    freeWorld();

    h_spheres = std::move(savedSpheres);
    h_triangles = std::move(savedTriangles);
    h_meshes = std::move(savedMeshes);
    h_bvh_nodes = std::move(savedBvhNodes);

    rebuildGPUWorld();
}
