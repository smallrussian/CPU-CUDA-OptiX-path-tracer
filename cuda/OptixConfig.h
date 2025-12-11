#ifndef OPTIX_CONFIG_H
#define OPTIX_CONFIG_H

// ============================================================================
// OptiX Ray Tracing Configuration
// ============================================================================
// SINGLE SOURCE OF TRUTH for all OptiX pipeline constants
// Changing values here automatically updates both host and device code
// This prevents CUDA error 700 (illegal memory access) from hardcoded mismatches
// ============================================================================

// Maximum ray bounce depth in shaders (closesthit recursion)
// Used in: optix_programs.cu (__closesthit__ch)
#define OPTIX_MAX_DEPTH 3

// Number of payload values per ray trace
// Current: 4 (r, g, b, hit_distance) - hit_distance for motion vectors
// CRITICAL: ALL optixTrace calls must use this exact count
// Used in: OptixRenderer.cu (createModule, createPipeline)
#define OPTIX_NUM_PAYLOADS 4

// Maximum trace depth for pipeline stack calculation
// Formula: MAX_DEPTH + 1 (extra level for shadow rays from primary hits)
// Used in: OptixRenderer.cu (createPipeline, stack size calculation)
#define OPTIX_MAX_TRACE_DEPTH (OPTIX_MAX_DEPTH + 1)

// Number of attribute values for intersection programs
// Spheres use 2 attributes (U, V coordinates on sphere surface)
#define OPTIX_NUM_ATTRIBUTES 2

// Default samples per pixel for anti-aliasing
// Can be overridden at runtime via render() parameter
// Used in: OptixRenderer.cu initialization
#define OPTIX_DEFAULT_SAMPLES_PER_PIXEL 1

#endif // OPTIX_CONFIG_H
