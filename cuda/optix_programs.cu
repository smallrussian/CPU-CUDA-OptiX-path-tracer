#include <optix.h>
#include <cuda_runtime.h>
#include "OptixConfig.h"

// Vector math helpers for float3
static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float3 operator/(float3 a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

static __forceinline__ __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

static __forceinline__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float length(float3 v) {
    return sqrtf(dot(v, v));
}

static __forceinline__ __device__ float3 normalize(float3 v) {
    float len = length(v);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

static __forceinline__ __device__ float3 reflect(float3 v, float3 n) {
    return v - 2.0f * dot(v, n) * n;
}

static __forceinline__ __device__ float3 refract(float3 uv, float3 n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    float3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float3 r_out_parallel = -sqrtf(fabsf(1.0f - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

// Schlick's approximation for reflectance
static __forceinline__ __device__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// Simple random number generator
static __forceinline__ __device__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0, v1 = val1, s0 = 0;
    for (unsigned int n = 0; n < 4; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

static __forceinline__ __device__ float rnd(unsigned int& seed) {
    seed = tea(seed, seed);
    return (seed & 0x00FFFFFF) / (float)0x01000000;
}

static __forceinline__ __device__ float3 random_in_unit_sphere(unsigned int& seed) {
    float3 p;
    do {
        p = make_float3(2.0f * rnd(seed) - 1.0f, 2.0f * rnd(seed) - 1.0f, 2.0f * rnd(seed) - 1.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

static __forceinline__ __device__ float3 random_unit_vector(unsigned int& seed) {
    return normalize(random_in_unit_sphere(seed));
}

static __forceinline__ __device__ float3 random_in_unit_disk(unsigned int& seed) {
    float3 p;
    do {
        p = make_float3(2.0f * rnd(seed) - 1.0f, 2.0f * rnd(seed) - 1.0f, 0.0f);
    } while (dot(p, p) >= 1.0f);
    return p;
}

// Material data structure - must match OptixMaterialData in OptixRenderer.h
struct MaterialData {
    int type;           // 0=lambertian, 1=metal, 2=dielectric
    float3 albedo;
    float fuzz_or_ior;
};

// Light data structure - must match OptixLightData in OptixRenderer.h
struct LightData {
    int type;           // 0=point, 1=distant, 2=sphere, 3=rect
    float3 position;
    float3 direction;
    float3 color;
    float intensity;
    float radius;
};

// Launch parameters - must match OptixParams in OptixRenderer.h
extern "C" {
__constant__ struct {
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
    LightData* lights;
    int num_lights;
    unsigned int frame_number;
    // Samples per pixel (from viewport settings)
    int samples_per_pixel;
    // Flow buffer enabled flag (prevents writing to null buffer)
    bool useFlowBuffer;
} params;
}

// SBT data for hitgroup - must match HitGroupRecord (without header)
struct HitGroupData {
    MaterialData material;
};

// Payload for ray data
static __forceinline__ __device__ void setPayload(float3 color, float t_hit) {
    optixSetPayload_0(__float_as_uint(color.x));
    optixSetPayload_1(__float_as_uint(color.y));
    optixSetPayload_2(__float_as_uint(color.z));
    optixSetPayload_3(__float_as_uint(t_hit));
}

static __forceinline__ __device__ float4 getPayload() {
    return make_float4(
        __uint_as_float(optixGetPayload_0()),
        __uint_as_float(optixGetPayload_1()),
        __uint_as_float(optixGetPayload_2()),
        __uint_as_float(optixGetPayload_3())
    );
}

// Trace shadow ray - uses miss index 1 for shadow miss program
// Returns true if occluded (something hit), false if clear (miss)
static __forceinline__ __device__ bool traceOcclusion(
    OptixTraversableHandle handle,
    float3 origin,
    float3 direction,
    float tmin,
    float tmax)
{
    // Initialize to 1 (occluded). If we hit something, closesthit sets occluded=1.
    // If we miss (nothing hit), shadow miss program sets occluded=0.
    unsigned int occluded = 1;
    optixTrace(
        handle,
        origin,
        direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0,  // SBT offset
        1,  // SBT stride
        1,  // missSBTIndex = 1 (shadow miss program)
        occluded
    );
    return occluded != 0;
}

// Sample direct lighting from all lights with shadow rays
static __forceinline__ __device__ float3 sampleDirectLighting(
    float3 hit_point,
    float3 normal,
    unsigned int& seed)
{
    float3 direct_light = make_float3(0.0f, 0.0f, 0.0f);

    // Safety check for null lights pointer
    if (params.lights == nullptr || params.num_lights <= 0) {
        return direct_light;
    }

    for (int i = 0; i < params.num_lights; i++) {
        LightData light = params.lights[i];
        float3 light_dir;
        float light_dist;
        float3 light_contribution;

        if (light.type == 0) {
            // Point light
            float3 to_light = light.position - hit_point;
            light_dist = length(to_light);
            light_dir = to_light / light_dist;
            float falloff = 1.0f / (light_dist * light_dist);
            light_contribution = light.color * light.intensity * 0.1f * falloff;
        } else if (light.type == 1) {
            // Distant light (sun) - matches CUDA intensity scaling
            // Direction should point FROM surface TO light (upward for sun)
            light_dir = normalize(light.direction);
            light_dist = 1e16f;
            light_contribution = light.color * light.intensity * 0.01f;
        } else if (light.type == 2) {
            // Sphere light - sample random point on sphere
            float3 random_offset = random_unit_vector(seed) * light.radius;
            float3 light_point = light.position + random_offset;
            float3 to_light = light_point - hit_point;
            light_dist = length(to_light);
            light_dir = to_light / light_dist;
            float falloff = 1.0f / (light_dist * light_dist);
            light_contribution = light.color * light.intensity * 0.1f * falloff;
        } else {
            continue;  // Unsupported light type
        }

        // N dot L - for top-facing surfaces with upward light, this should be positive
        float NdotL = fmaxf(0.0f, dot(normal, light_dir));
        if (NdotL <= 0.0f) continue;

        // Shadow ray test
        bool in_shadow = traceOcclusion(
            params.traversable,
            hit_point + normal * 0.001f,
            light_dir,
            0.001f,
            light_dist - 0.001f
        );

        if (!in_shadow) {
            direct_light = direct_light + light_contribution * NdotL;
        }
    }

    return direct_light;
}

// Compute motion vector (optical flow) for temporal denoising
// Returns pixel displacement from previous frame to current frame
static __device__ float2 computeMotionVector(
    float3 ray_origin,
    float3 ray_direction,
    float t_hit,
    float3 cam_eye,
    float3 cam_u,
    float3 cam_v,
    float3 cam_w,
    float3 prev_cam_eye,
    float3 prev_cam_u,
    float3 prev_cam_v,
    float3 prev_cam_w,
    int width,
    int height
) {
    // Edge case 1: Sky hit (infinite distance)
    if (t_hit >= 1e15f) {
        return make_float2(0.0f, 0.0f);
    }

    // Edge case 2: First frame (previous camera not initialized)
    if (length(prev_cam_eye) == 0.0f) {
        return make_float2(0.0f, 0.0f);
    }

    // 1. Reconstruct 3D world point from ray and hit distance
    float3 world_point = ray_origin + t_hit * ray_direction;

    // 2. Project world point to current camera screen space
    // The ray generation uses: focal_point = cam_eye + cam_w + (2*u-1) * cam_u + (2*v-1) * cam_v
    // So to reverse it, we need to express world_point in terms of the camera basis

    float3 rel = world_point - cam_eye;

    // Get the component along each camera axis
    // cam_w is forward (to focal plane center), cam_u is right, cam_v is up
    float3 cam_w_norm = normalize(cam_w);
    float3 cam_u_norm = normalize(cam_u);
    float3 cam_v_norm = normalize(cam_v);

    float depth = dot(rel, cam_w_norm);

    // Edge case 3: Point behind camera
    if (depth <= 0.0f) {
        return make_float2(0.0f, 0.0f);
    }

    // Project rel onto the u and v axes
    float u_comp = dot(rel, cam_u_norm);
    float v_comp = dot(rel, cam_v_norm);

    // Apply perspective projection: scale by focal distance / depth
    // This accounts for the fact that points at different depths project differently
    float focal_dist = length(cam_w);
    float proj_u = (u_comp * focal_dist) / depth;
    float proj_v = (v_comp * focal_dist) / depth;

    // Scale by the camera basis lengths to get normalized coordinates
    // cam_u and cam_v are scaled by half viewport dimensions
    float u_normalized = proj_u / length(cam_u);  // In range approximately [-1, 1]
    float v_normalized = proj_v / length(cam_v);

    // Convert from [-1, 1] to [0, 1] to [0, width/height]
    float curr_pixel_x = (u_normalized * 0.5f + 0.5f) * width;
    float curr_pixel_y = (v_normalized * 0.5f + 0.5f) * height;

    // 3. Project same world point to previous camera screen space
    float3 prev_rel = world_point - prev_cam_eye;

    float3 prev_cam_w_norm = normalize(prev_cam_w);
    float3 prev_cam_u_norm = normalize(prev_cam_u);
    float3 prev_cam_v_norm = normalize(prev_cam_v);

    float prev_depth = dot(prev_rel, prev_cam_w_norm);

    // Edge case 4: Point was behind previous camera
    if (prev_depth <= 0.0f) {
        return make_float2(0.0f, 0.0f);
    }

    float prev_u_comp = dot(prev_rel, prev_cam_u_norm);
    float prev_v_comp = dot(prev_rel, prev_cam_v_norm);

    // Apply perspective projection for previous camera
    float prev_focal_dist = length(prev_cam_w);
    float prev_proj_u = (prev_u_comp * prev_focal_dist) / prev_depth;
    float prev_proj_v = (prev_v_comp * prev_focal_dist) / prev_depth;

    float prev_u_normalized = prev_proj_u / length(prev_cam_u);
    float prev_v_normalized = prev_proj_v / length(prev_cam_v);

    float prev_pixel_x = (prev_u_normalized * 0.5f + 0.5f) * width;
    float prev_pixel_y = (prev_v_normalized * 0.5f + 0.5f) * height;

    // 4. Compute optical flow: previous position - current position (backward flow)
    float flow_x = prev_pixel_x - curr_pixel_x;
    float flow_y = prev_pixel_y - curr_pixel_y;

    return make_float2(flow_x, flow_y);
}

// Ray generation program
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Initialize random seed
    unsigned int seed = tea(idx.x + idx.y * dim.x, params.frame_number);

    // Accumulate color over multiple samples
    float3 accumulated_color = make_float3(0.0f, 0.0f, 0.0f);

    // Declare payload variables outside loop for motion vector computation
    unsigned int p0, p1, p2, p3;

    for (int s = 0; s < params.samples_per_pixel; s++) {
        // Calculate normalized screen coordinates with jitter for anti-aliasing
        const float u = (float(idx.x) + rnd(seed)) / float(dim.x);
        const float v = (float(idx.y) + rnd(seed)) / float(dim.y);

        // Calculate point on focal plane
        float3 focal_point = params.cam_eye + params.cam_w +
            (2.0f * u - 1.0f) * params.cam_u +
            (2.0f * v - 1.0f) * params.cam_v;

        // Depth of field: jitter ray origin on lens disk
        float3 ray_origin = params.cam_eye;
        if (params.lens_radius > 0.0f) {
            float3 rd = random_in_unit_disk(seed);
            float3 offset = params.defocus_disk_u * rd.x + params.defocus_disk_v * rd.y;
            ray_origin = params.cam_eye + offset;
        }

        // Ray direction from (possibly jittered) origin to focal point
        float3 ray_direction = normalize(focal_point - ray_origin);

        // Initialize payload with seed for random in hit programs
        // Encode depth in high 4 bits of seed (0 = primary ray)
        p0 = __float_as_uint(0.0f);
        p1 = __float_as_uint(0.0f);
        p2 = (seed & 0x0FFFFFFF);  // Pass seed with depth=0 in high bits
        p3 = __uint_as_float(0xFFFFFFFF);  // Initialize to infinity (no hit)

        // Trace ray
        optixTrace(
            params.traversable,
            ray_origin,
            ray_direction,
            0.001f,              // tmin
            1e16f,               // tmax
            0.0f,                // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            1,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2, p3
        );

        // Accumulate sample color
        accumulated_color.x += __uint_as_float(p0);
        accumulated_color.y += __uint_as_float(p1);
        accumulated_color.z += __uint_as_float(p2);
    }

    // Average samples
    float3 color = accumulated_color / float(params.samples_per_pixel);

    // Compute motion vector for temporal denoising (only if buffer is enabled)
    if (params.useFlowBuffer) {
        // Use hit distance from last sample's p3 payload
        float last_hit_distance = __uint_as_float(p3);

        // Calculate center ray for motion vector (no jitter, no depth of field)
        const float center_u = (float(idx.x) + 0.5f) / float(dim.x);
        const float center_v = (float(idx.y) + 0.5f) / float(dim.y);

        float3 focal_point = params.cam_eye + params.cam_w +
            (2.0f * center_u - 1.0f) * params.cam_u +
            (2.0f * center_v - 1.0f) * params.cam_v;

        float3 center_ray_direction = normalize(focal_point - params.cam_eye);

        // Compute motion vector using render resolution (not output resolution!)
        float2 motion_vector = computeMotionVector(
            params.cam_eye,
            center_ray_direction,
            last_hit_distance,
            params.cam_eye,
            params.cam_u,
            params.cam_v,
            params.cam_w,
            params.prev_cam_eye,
            params.prev_cam_u,
            params.prev_cam_v,
            params.prev_cam_w,
            params.renderWidth,
            params.renderHeight
        );

        // Write motion vector to flow buffer (use renderWidth for stride)
        params.flowBuffer[idx.y * params.renderWidth + idx.x] = motion_vector;
    }

    // Gamma correction (gamma 2.0 = sqrt)
    color.x = sqrtf(fmaxf(0.0f, color.x));
    color.y = sqrtf(fmaxf(0.0f, color.y));
    color.z = sqrtf(fmaxf(0.0f, color.z));

    // Write to framebuffer (use renderWidth for stride - matches actual buffer dimensions)
    params.framebuffer[idx.y * params.renderWidth + idx.x] = make_float4(color.x, color.y, color.z, 1.0f);
}

// Miss program - sky gradient (for primary rays)
extern "C" __global__ void __miss__ms() {
    const float3 ray_dir = optixGetWorldRayDirection();

    // Sky gradient based on ray Y direction
    float t = 0.5f * (ray_dir.y + 1.0f);
    float3 white = make_float3(1.0f, 1.0f, 1.0f);
    float3 blue = make_float3(0.5f, 0.7f, 1.0f);

    float3 color = (1.0f - t) * white + t * blue;

    // Check for sun disk (distant lights) - with null safety
    if (params.lights != nullptr && params.num_lights > 0) {
        for (int i = 0; i < params.num_lights; i++) {
            if (params.lights[i].type == 1) {  // Distant light
            float3 sun_dir = normalize(params.lights[i].direction);
            float sun_dot = dot(ray_dir, sun_dir);
            if (sun_dot > 0.995f) {  // ~5 degree sun disk
                float intensity = powf((sun_dot - 0.995f) / 0.005f, 4.0f);
                color = color + params.lights[i].color * intensity * 2.0f;
            }
            }
        }
    }

    setPayload(color, 1e16f);  // Sky has infinite distance
}

// Shadow miss program - called when shadow ray doesn't hit anything
extern "C" __global__ void __miss__shadow() {
    // Set occlusion to 0 (not occluded - ray reached the light)
    optixSetPayload_0(0);
}

// Closest hit program - sphere shading with materials
extern "C" __global__ void __closesthit__ch() {
    // Get hit info
    const float t_hit = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    // Compute hit point
    float3 hit_point = ray_origin + t_hit * ray_dir;

    // Get sphere data and compute normal
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGASIndex = optixGetSbtGASIndex();

    float4 sphere_data;
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.0f, &sphere_data);

    float3 center = make_float3(sphere_data.x, sphere_data.y, sphere_data.z);
    float radius = sphere_data.w;
    float3 normal = normalize((hit_point - center) / radius);

    // Ensure normal faces the ray
    bool front_face = dot(ray_dir, normal) < 0.0f;
    if (!front_face) normal = -normal;

    // Get material from SBT record
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    MaterialData mat = sbt_data->material;

    // Get random seed and depth from payload
    // Depth is encoded in high 4 bits of payload_2
    unsigned int payload2 = optixGetPayload_2();
    unsigned int seed = payload2 & 0x0FFFFFFF;
    unsigned int depth = (payload2 >> 28) & 0xF;

    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // Simple directional light for base shading
    float3 light_dir = normalize(make_float3(1.0f, 1.0f, 1.0f));
    float NdotL = fmaxf(0.0f, dot(normal, light_dir));
    float3 ambient = make_float3(0.15f, 0.15f, 0.15f);
    float3 diffuse = make_float3(NdotL * 0.85f, NdotL * 0.85f, NdotL * 0.85f);

    if (mat.type == 0) {
        // Lambertian - diffuse material with direct + indirect lighting
        float3 albedo = mat.albedo;

        // Sample direct lighting on ALL bounces for proper color bleeding (matches CUDA)
        float3 direct = sampleDirectLighting(hit_point, normal, seed);

        if (depth < OPTIX_MAX_DEPTH) {
            // Scatter ray in random direction (cosine-weighted hemisphere)
            float3 scatter_dir = normalize(normal + random_unit_vector(seed));

            // Trace bounce ray for indirect lighting
            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);  // Initialize to infinity

            optixTrace(
                params.traversable,
                hit_point + normal * 0.001f,
                scatter_dir,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            float3 indirect = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
            // Combine direct lighting (with shadows) + indirect bounces
            color = albedo * (direct + indirect);
        } else {
            // Max depth - return ambient + direct
            color = albedo * (ambient + direct);
        }

    } else if (mat.type == 1) {
        // Metal - reflections with direct lighting
        float3 albedo = mat.albedo;
        float fuzz = fminf(mat.fuzz_or_ior, 1.0f);

        // Sample direct lighting on ALL bounces (matches CUDA)
        // Direct light is NOT scaled by fuzz - fuzz only affects reflection scatter
        float3 direct = sampleDirectLighting(hit_point, normal, seed);

        // Compute reflection direction
        float3 reflected = reflect(normalize(ray_dir), normal);
        reflected = normalize(reflected + fuzz * random_in_unit_sphere(seed));

        // Only trace if reflection is in correct hemisphere AND we haven't exceeded max depth
        if (dot(reflected, normal) > 0.0f && depth < OPTIX_MAX_DEPTH) {
            // Trace reflection ray with incremented depth
            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            // Encode incremented depth in high bits
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);  // Initialize to infinity

            optixTrace(
                params.traversable,
                hit_point + normal * 0.001f,
                reflected,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            float3 reflected_color = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
            // Direct light added at full strength (matches CUDA behavior)
            color = albedo * (reflected_color + direct);
        } else {
            // Max depth reached or invalid reflection - use direct lighting with shadows
            color = albedo * (ambient + direct);
        }

    } else if (mat.type == 2) {
        // Dielectric - glass material with refraction
        if (depth < OPTIX_MAX_DEPTH) {
            float ior = mat.fuzz_or_ior;
            float refraction_ratio = front_face ? (1.0f / ior) : ior;

            float3 unit_direction = normalize(ray_dir);
            float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            float3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rnd(seed)) {
                // Total internal reflection
                direction = reflect(unit_direction, normal);
            } else {
                // Refract
                direction = refract(unit_direction, normal, refraction_ratio);
            }

            // Trace refraction/reflection ray with incremented depth
            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);  // Initialize to infinity

            optixTrace(
                params.traversable,
                hit_point + direction * 0.001f,
                direction,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            color = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
        } else {
            // Max depth - return sky color approximation
            color = make_float3(0.5f, 0.7f, 1.0f);
        }
    } else {
        // Unknown material type - use diffuse
        color = mat.albedo * (ambient + diffuse);
    }

    // Clamp color
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));

    setPayload(color, t_hit);  // Pass hit distance for motion vectors
}

// Closest hit program for triangles - uses built-in triangle intersection
extern "C" __global__ void __closesthit__triangle() {
    // Get hit info
    const float t_hit = optixGetRayTmax();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    // Compute hit point
    float3 hit_point = ray_origin + t_hit * ray_dir;

    // Get triangle vertices using built-in function (OptiX 9.0 API)
    float3 verts[3];
    optixGetTriangleVertexData(verts);
    float3 v0 = verts[0];
    float3 v1 = verts[1];
    float3 v2 = verts[2];

    // Compute flat normal from triangle edges using inline cross product
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 normal = make_float3(
        edge1.y * edge2.z - edge1.z * edge2.y,
        edge1.z * edge2.x - edge1.x * edge2.z,
        edge1.x * edge2.y - edge1.y * edge2.x
    );
    normal = normalize(normal);

    // Ensure normal faces the ray
    bool front_face = dot(ray_dir, normal) < 0.0f;
    if (!front_face) normal = -normal;

    // Get material from SBT record
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    MaterialData mat = sbt_data->material;

    // Get random seed and depth from payload (same encoding as sphere shader)
    // Depth is encoded in high 4 bits of payload_2, seed in low 28 bits
    unsigned int payload2 = optixGetPayload_2();
    unsigned int seed = payload2 & 0x0FFFFFFF;
    unsigned int depth = (payload2 >> 28) & 0xF;

    // === MATERIAL SHADING (same pattern as sphere) ===
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    // Simple directional light for base shading (matches sphere shader)
    float3 light_dir = normalize(make_float3(1.0f, 1.0f, 1.0f));
    float NdotL = fmaxf(0.0f, dot(normal, light_dir));
    float3 ambient = make_float3(0.15f, 0.15f, 0.15f);
    float3 diffuse = make_float3(NdotL * 0.85f, NdotL * 0.85f, NdotL * 0.85f);

    if (mat.type == 0) {
        // Lambertian - diffuse material with direct + indirect lighting
        float3 albedo = mat.albedo;

        // Sample direct lighting on ALL bounces for proper color bleeding (matches CUDA)
        float3 direct = sampleDirectLighting(hit_point, normal, seed);

        if (depth < OPTIX_MAX_DEPTH) {
            // Scatter ray in random direction
            float3 scatter_dir = normalize(normal + random_unit_vector(seed));

            // Trace bounce ray
            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);

            optixTrace(
                params.traversable,
                hit_point + normal * 0.001f,
                scatter_dir,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            float3 indirect = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
            color = albedo * (direct + indirect);
        } else {
            color = albedo * (ambient + direct);
        }

    } else if (mat.type == 1) {
        // Metal - reflections with direct lighting
        float3 albedo = mat.albedo;
        float fuzz = fminf(mat.fuzz_or_ior, 1.0f);

        // Sample direct lighting on ALL bounces (matches CUDA)
        // Direct light is NOT scaled by fuzz - fuzz only affects reflection scatter
        float3 direct = sampleDirectLighting(hit_point, normal, seed);

        float3 reflected = reflect(normalize(ray_dir), normal);
        reflected = normalize(reflected + fuzz * random_in_unit_sphere(seed));

        if (dot(reflected, normal) > 0.0f && depth < OPTIX_MAX_DEPTH) {
            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);

            optixTrace(
                params.traversable,
                hit_point + normal * 0.001f,
                reflected,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            float3 reflected_color = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
            // Direct light added at full strength (matches CUDA behavior)
            color = albedo * (reflected_color + direct);
        } else {
            color = albedo * (ambient + direct);
        }

    } else if (mat.type == 2) {
        // Dielectric - glass with refraction
        if (depth < OPTIX_MAX_DEPTH) {
            float ior = mat.fuzz_or_ior;
            float refraction_ratio = front_face ? (1.0f / ior) : ior;

            float3 unit_direction = normalize(ray_dir);
            float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
            float3 direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > rnd(seed)) {
                direction = reflect(unit_direction, normal);
            } else {
                direction = refract(unit_direction, normal, refraction_ratio);
            }

            unsigned int r0, r1, r2, r3;
            r0 = __float_as_uint(0.0f);
            r1 = __float_as_uint(0.0f);
            r2 = (seed & 0x0FFFFFFF) | ((depth + 1) << 28);
            r3 = __uint_as_float(0xFFFFFFFF);

            optixTrace(
                params.traversable,
                hit_point + direction * 0.001f,
                direction,
                0.001f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0, 1, 0,
                r0, r1, r2, r3
            );

            color = make_float3(__uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2));
        } else {
            color = make_float3(0.5f, 0.7f, 1.0f);
        }
    } else {
        // Unknown material type
        color = mat.albedo * (ambient + diffuse);
    }

    // Clamp color
    color.x = fminf(1.0f, fmaxf(0.0f, color.x));
    color.y = fminf(1.0f, fmaxf(0.0f, color.y));
    color.z = fminf(1.0f, fmaxf(0.0f, color.z));

    setPayload(color, t_hit);
}

