#ifndef CUDA_TRIANGLE_H
#define CUDA_TRIANGLE_H

#include "hitable.h"
#include "aabb.h"

class triangle : public hitable {
public:
    __device__ triangle() {}
    __device__ triangle(vec3 v0, vec3 v1, vec3 v2, material* m)
        : v0(v0), v1(v1), v2(v2), mat_ptr(m), use_vertex_normals(false)
    {
        // Compute flat shading normal from triangle edges
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
    }

    __device__ triangle(vec3 v0, vec3 v1, vec3 v2,
                        vec3 n0, vec3 n1, vec3 n2,
                        material* m)
        : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2),
          mat_ptr(m), use_vertex_normals(true)
    {
        // Compute geometric normal for backface detection
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ bool bounding_box(aabb& box) const;

    vec3 v0, v1, v2;           // Vertices
    vec3 n0, n1, n2;           // Per-vertex normals (optional)
    vec3 normal;               // Geometric/flat normal
    material* mat_ptr;
    bool use_vertex_normals;
};

// Möller-Trumbore ray-triangle intersection algorithm
__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    const float EPSILON = 1e-7f;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;

    // Calculate determinant
    vec3 h = cross(r.direction(), edge2);
    float det = dot(edge1, h);

    // Ray is parallel to triangle
    if (fabsf(det) < EPSILON)
        return false;

    float inv_det = 1.0f / det;

    // Calculate u parameter (barycentric coordinate)
    vec3 s = r.origin() - v0;
    float u = inv_det * dot(s, h);

    // Check u bounds
    if (u < 0.0f || u > 1.0f)
        return false;

    // Calculate v parameter
    vec3 q = cross(s, edge1);
    float v = inv_det * dot(r.direction(), q);

    // Check v bounds and u+v <= 1
    if (v < 0.0f || u + v > 1.0f)
        return false;

    // Calculate t (distance along ray)
    float t = inv_det * dot(edge2, q);

    // Check if t is within valid interval
    if (t < t_min || t > t_max)
        return false;

    // Valid hit - fill in the record
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    rec.mat_ptr = mat_ptr;

    // Compute normal (interpolated or flat)
    if (use_vertex_normals) {
        // Barycentric interpolation: w = 1 - u - v
        float w = 1.0f - u - v;
        rec.normal = unit_vector(n0 * w + n1 * u + n2 * v);
    } else {
        rec.normal = normal;
    }

    // Handle backface (flip normal if ray hits from behind)
    if (dot(r.direction(), rec.normal) > 0.0f) {
        rec.normal = -rec.normal;
    }

    return true;
}

__device__ bool triangle::bounding_box(aabb& box) const {
    // Find min and max for each axis across all 3 vertices
    // Add small padding to avoid degenerate boxes for axis-aligned triangles
    const float pad = 0.0001f;

    vec3 min_pt(
        fminf(fminf(v0.x(), v1.x()), v2.x()) - pad,
        fminf(fminf(v0.y(), v1.y()), v2.y()) - pad,
        fminf(fminf(v0.z(), v1.z()), v2.z()) - pad
    );
    vec3 max_pt(
        fmaxf(fmaxf(v0.x(), v1.x()), v2.x()) + pad,
        fmaxf(fmaxf(v0.y(), v1.y()), v2.y()) + pad,
        fmaxf(fmaxf(v0.z(), v1.z()), v2.z()) + pad
    );

    box = aabb(min_pt, max_pt);
    return true;
}

#endif
