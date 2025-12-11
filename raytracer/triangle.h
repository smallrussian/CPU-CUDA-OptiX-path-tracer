#ifndef RT_TRIANGLE_H
#define RT_TRIANGLE_H
//==============================================================================================
// Triangle primitive for ray tracing
// Uses Möller-Trumbore algorithm for ray-triangle intersection
// RTiOW-style implementation (non-templated, inherits from hittable)
//==============================================================================================

#include "hittable.h"
#include "aabb.h"

namespace rt {

class triangle : public hittable {
  public:
    triangle(const point3& v0, const point3& v1, const point3& v2, shared_ptr<material> mat)
      : v0(v0), v1(v1), v2(v2), mat(mat)
    {
        // Compute flat shading normal from triangle edges
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
    }

    triangle(const point3& v0, const point3& v1, const point3& v2,
             const vec3& n0, const vec3& n1, const vec3& n2,
             shared_ptr<material> mat)
      : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), mat(mat), use_vertex_normals(true)
    {
        // Compute geometric normal for backface detection
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        // Möller-Trumbore algorithm
        const double EPSILON = 1e-8;

        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;

        // Calculate determinant
        vec3 h = cross(r.direction(), edge2);
        double det = dot(edge1, h);

        // Ray is parallel to triangle
        if (std::fabs(det) < EPSILON)
            return false;

        double inv_det = 1.0 / det;

        // Calculate u parameter
        vec3 s = r.origin() - v0;
        double u = inv_det * dot(s, h);

        // Check u bounds
        if (u < 0.0 || u > 1.0)
            return false;

        // Calculate v parameter
        vec3 q = cross(s, edge1);
        double v = inv_det * dot(r.direction(), q);

        // Check v bounds and u+v <= 1
        if (v < 0.0 || u + v > 1.0)
            return false;

        // Calculate t (distance along ray)
        double t = inv_det * dot(edge2, q);

        // Check if t is within valid interval
        if (!ray_t.surrounds(t))
            return false;

        // Valid hit - fill in the record
        rec.t = t;
        rec.p = r.at(t);
        rec.mat = mat;

        // Compute normal (interpolated or flat)
        vec3 outward_normal;
        if (use_vertex_normals) {
            // Barycentric interpolation: w = 1 - u - v
            double w = 1.0 - u - v;
            outward_normal = unit_vector(n0 * w + n1 * u + n2 * v);
        } else {
            outward_normal = normal;
        }

        rec.set_face_normal(r, outward_normal);
        return true;
    }

    aabb bounding_box() const {
        // Find min and max for each axis across all 3 vertices
        point3 min_pt(
            std::fmin(std::fmin(v0[0], v1[0]), v2[0]),
            std::fmin(std::fmin(v0[1], v1[1]), v2[1]),
            std::fmin(std::fmin(v0[2], v1[2]), v2[2])
        );
        point3 max_pt(
            std::fmax(std::fmax(v0[0], v1[0]), v2[0]),
            std::fmax(std::fmax(v0[1], v1[1]), v2[1]),
            std::fmax(std::fmax(v0[2], v1[2]), v2[2])
        );
        return aabb(min_pt, max_pt);
    }

  private:
    point3 v0, v1, v2;           // Vertices
    vec3 n0, n1, n2;             // Per-vertex normals (optional)
    vec3 normal;                  // Geometric/flat normal
    shared_ptr<material> mat;
    bool use_vertex_normals = false;
};

} // namespace rt

#endif
