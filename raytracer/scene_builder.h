#ifndef RT_SCENE_BUILDER_H
#define RT_SCENE_BUILDER_H

#include "rtweekend.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "triangle.h"
#include "bvh.h"

#include "../graphics/Mesh.h"
#include "../graphics/Light.h"

#include <vector>
#include <iostream>

namespace rt {

// Material info struct to pass material data from scene parsing
struct MaterialInfo {
    color albedo = color(0.5, 0.5, 0.5);  // Base color
    double roughness = 1.0;                // 0 = mirror, 1 = diffuse
    double metallic = 0.0;                 // 0 = dielectric, 1 = metal
    double ior = 1.5;                      // Index of refraction (for glass)
    bool is_glass = false;                 // True for dielectric materials
};


class scene_builder {
  public:
    scene_builder() = default;

    // Add a mesh to the scene with material info
    // Uses flat shading (geometric normals) for correct lighting on hard-edged geometry
    void add_mesh(const graphics::Mesh& mesh, const MaterialInfo& mat_info) {
        auto mat = create_material(mat_info);

        const std::vector<graphics::Vertex>& vertices = mesh.getVertices();
        const std::vector<graphics::TriangleFace>& faces = mesh.getFaces();
        const graphics::Transformationf& transform = mesh.getTransform();

        std::clog << "[scene_builder::add_mesh] Adding mesh with " << vertices.size()
                  << " vertices, " << faces.size() << " faces" << std::endl;

        // Get rotation matrix for transforming normals
        graphics::Matrix3<float> rotMatrix = transform.getRotation().toRotationMatrix();

        // FIRST PASS: Transform all vertices and compute mesh centroid
        std::vector<point3> transformed_verts;
        transformed_verts.reserve(vertices.size());
        point3 mesh_centroid(0, 0, 0);

        for (const auto& v : vertices) {
            point3 tv = transform_point(v.position, transform, rotMatrix);
            transformed_verts.push_back(tv);
            mesh_centroid = mesh_centroid + tv;
        }
        mesh_centroid = mesh_centroid / static_cast<double>(vertices.size());

        // SECOND PASS: Create triangles with correctly oriented normals
        for (const auto& face : faces) {
            unsigned int i0 = face.indices[graphics::A];
            unsigned int i1 = face.indices[graphics::B];
            unsigned int i2 = face.indices[graphics::C];

            // Use pre-transformed vertices
            point3 v0 = transformed_verts[i0];
            point3 v1 = transformed_verts[i1];
            point3 v2 = transformed_verts[i2];

            // Compute flat (geometric) normal from triangle edges
            vec3 edge1 = v1 - v0;
            vec3 edge2 = v2 - v0;
            vec3 flat_normal = unit_vector(cross(edge1, edge2));

            // Ensure normal points outward (away from actual mesh centroid)
            point3 tri_center = (v0 + v1 + v2) / 3.0;
            vec3 to_triangle = tri_center - mesh_centroid;
            if (dot(flat_normal, to_triangle) < 0) {
                flat_normal = -flat_normal;
            }

            // Create triangle with flat shading (same normal for all vertices)
            objects.add(make_shared<triangle>(v0, v1, v2, flat_normal, flat_normal, flat_normal, mat));
        }
    }

    // Add a sphere primitive to the scene (RTiOW style - mathematical sphere)
    void add_sphere(const point3& center, double radius, const MaterialInfo& mat_info) {
        auto mat = create_material(mat_info);
        std::clog << "[scene_builder::add_sphere] Adding sphere at ("
                  << center[0] << "," << center[1] << "," << center[2]
                  << ") radius=" << radius << std::endl;
        objects.add(make_shared<sphere>(center, radius, mat));
        std::clog << "[scene_builder::add_sphere] Total objects: " << objects.objects.size() << std::endl;
    }

    // Add a triangle primitive directly (for converting from CUDA triangle data)
    void add_triangle(const point3& v0, const point3& v1, const point3& v2,
                      const vec3& n0, const vec3& n1, const vec3& n2,
                      bool use_vertex_normals, const MaterialInfo& mat_info) {
        auto mat = create_material(mat_info);
        if (use_vertex_normals) {
            objects.add(make_shared<triangle>(v0, v1, v2, n0, n1, n2, mat));
        } else {
            // Compute flat normal from triangle edges
            vec3 edge1 = v1 - v0;
            vec3 edge2 = v2 - v0;
            vec3 flat_normal = unit_vector(cross(edge1, edge2));
            objects.add(make_shared<triangle>(v0, v1, v2, flat_normal, flat_normal, flat_normal, mat));
        }
    }

    // Add a light to the scene
    void add_light(const graphics::Lightf& light) {
        lights.push_back(light);
    }

    // Build BVH and return the scene root
    shared_ptr<hittable> build_bvh() {
        if (objects.objects.empty()) {
            return nullptr;
        }
        std::clog << "[scene_builder] Building BVH with " << objects.objects.size() << " objects" << std::endl;
        auto bvh = make_shared<bvh_node>(objects);
        std::clog << "[scene_builder] BVH built successfully" << std::endl;
        return bvh;
    }

    // Build without BVH (returns hittable_list directly)
    shared_ptr<hittable> build_list() {
        std::clog << "[scene_builder] Building scene list (no BVH) with " << objects.objects.size() << " objects" << std::endl;
        return make_shared<hittable_list>(objects);
    }

    // Get the hittable_list directly (for non-BVH rendering)
    const hittable_list& get_world() const {
        return objects;
    }

    // Get the lights
    const std::vector<graphics::Lightf>& get_lights() const {
        return lights;
    }

    // Clear all objects and lights
    void clear() {
        objects.clear();
        lights.clear();
    }

    // Get object count (for debugging)
    size_t get_object_count() const {
        return objects.objects.size();
    }

  private:
    // Create RTiOW material from MaterialInfo
    shared_ptr<material> create_material(const MaterialInfo& info) {
        if (info.is_glass) {
            // Dielectric (glass) material
            return make_shared<dielectric>(info.ior);
        } else if (info.metallic > 0.5) {
            // Metal material
            double fuzz = info.roughness;  // roughness maps to fuzz
            return make_shared<metal>(info.albedo, fuzz);
        } else {
            // Lambertian (diffuse) material
            return make_shared<lambertian>(info.albedo);
        }
    }

    // Transform a point by a Transformation (scale, rotate, translate)
    point3 transform_point(const graphics::Vector3f& point,
                           const graphics::Transformationf& transform,
                           const graphics::Matrix3<float>& rotMatrix) const {
        const graphics::Vector3f& scale = transform.getScale();
        const graphics::Vector3f& position = transform.getPosition();

        // Apply scale
        graphics::Vector3f scaled(
            point.x() * scale.x(),
            point.y() * scale.y(),
            point.z() * scale.z()
        );

        // Apply rotation using matrix multiplication
        graphics::Vector3f rotated = rotMatrix * scaled;

        // Apply translation
        return point3(
            static_cast<double>(rotated.x() + position.x()),
            static_cast<double>(rotated.y() + position.y()),
            static_cast<double>(rotated.z() + position.z())
        );
    }

    hittable_list objects;
    std::vector<graphics::Lightf> lights;
};

} // namespace rt

#endif
