
#include "ViewportGL.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <QTimer>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QPainter>
#include <QOpenGLVersionFunctionsFactory>
#include <algorithm>
#include <map>
#include "graphics/Vertex.h"
#include "graphics/Face.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to generate UV sphere mesh for GL rendering
namespace {
    void generateSphereMesh(std::vector<graphics::Vertex>& vertices,
                            std::vector<graphics::TriangleFace>& faces,
                            const graphics::Vector3f& center,
                            float radius,
                            int segments = 32,
                            int rings = 16) {
        using namespace graphics;

        vertices.clear();
        faces.clear();

        // Generate vertices
        for (int ring = 0; ring <= rings; ++ring) {
            float phi = M_PI * float(ring) / float(rings);  // 0 to PI
            float y = std::cos(phi);
            float ringRadius = std::sin(phi);

            for (int seg = 0; seg <= segments; ++seg) {
                float theta = 2.0f * M_PI * float(seg) / float(segments);  // 0 to 2PI
                float x = ringRadius * std::cos(theta);
                float z = ringRadius * std::sin(theta);

                Vertex v;
                v.normal = Vector3f(x, y, z);  // Normal is just the unit sphere position
                v.position = center + v.normal * radius;
                vertices.push_back(v);
            }
        }

        // Generate faces (quads split into triangles)
        for (int ring = 0; ring < rings; ++ring) {
            for (int seg = 0; seg < segments; ++seg) {
                int current = ring * (segments + 1) + seg;
                int next = current + segments + 1;

                // First triangle
                TriangleFace f1;
                f1.indices[0] = current;
                f1.indices[1] = next;
                f1.indices[2] = current + 1;
                faces.push_back(f1);

                // Second triangle
                TriangleFace f2;
                f2.indices[0] = current + 1;
                f2.indices[1] = next;
                f2.indices[2] = next + 1;
                faces.push_back(f2);
            }
        }
    }

    // Generate a flat ground plane for GL rendering (used instead of huge ground spheres)
    void generateGroundPlane(std::vector<graphics::Vertex>& vertices,
                             std::vector<graphics::TriangleFace>& faces,
                             float y_height,
                             float size = 100.0f) {
        using namespace graphics;

        vertices.clear();
        faces.clear();

        // Create a large quad centered at origin at the specified Y height
        float half = size;

        Vertex v0, v1, v2, v3;
        v0.position = Vector3f(-half, y_height, -half);
        v0.normal = Vector3f(0, 1, 0);

        v1.position = Vector3f(half, y_height, -half);
        v1.normal = Vector3f(0, 1, 0);

        v2.position = Vector3f(half, y_height, half);
        v2.normal = Vector3f(0, 1, 0);

        v3.position = Vector3f(-half, y_height, half);
        v3.normal = Vector3f(0, 1, 0);

        vertices.push_back(v0);
        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);

        // Two triangles for the quad
        TriangleFace f1, f2;
        f1.indices[0] = 0;
        f1.indices[1] = 2;
        f1.indices[2] = 1;

        f2.indices[0] = 0;
        f2.indices[1] = 3;
        f2.indices[2] = 2;

        faces.push_back(f1);
        faces.push_back(f2);
    }
}

namespace graphics {

// Removed spinning light variables - now using scene lights from USDA

ViewportGL::ViewportGL(QWidget *parent) : QOpenGLWidget(parent) {
	this->setMouseTracking(true);
	this->setFocusPolicy(Qt::StrongFocus);  // Enable keyboard focus

	this->camera = std::make_unique<MouseCameraf>(8.0f);  // Zoomed out
	this->camera->setRotation(0.5f, 1.3f);  // Orbited right, level view with slight downward tilt
	this->camera->setFarPlane(5000.0f);  // Increased for large ground sphere

	// Initialize ray tracing members
	this->rayTracer = std::make_unique<Renderer>();

	// Initialize render settings
	this->samplesPerPixel = 10;  // Higher for better quality, less noise
	this->maxDepth = 10;

	// Initialize scene state
	this->sceneLoaded = false;

    this->timer = new QTimer(this);
    this->timer->setInterval(16);
    connect(this->timer, &QTimer::timeout, this, &ViewportGL::onTimeout);
}

ViewportGL::~ViewportGL() {

}

void ViewportGL::onTimeout() {
	// Light no longer spins - uses scene lights from USDA
    this->update();
}

void ViewportGL::initializeGL() {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());
	f->initializeOpenGLFunctions();

    f->glClearColor(0.75f, 0.85f, 1.0f, 1.0f);  // Sky blue background
    f->glEnable(GL_DEPTH_TEST);

	// Load pending scene file if set (before viewport shows)
	if (!pendingSceneFile.empty()) {
		std::cout << "[Viewport] Loading pending scene: " << pendingSceneFile << std::endl;
		loadScene(pendingSceneFile);
		pendingSceneFile.clear();
		// Note: sceneLoadComplete signal is emitted inside loadUSD/loadOBJ
	}

	// Ensure we can receive keyboard input
	this->setFocus();

    this->timer->start();
}

void ViewportGL::setPendingSceneFile(const std::string& filename) {
	pendingSceneFile = filename;
	std::cout << "[Viewport] Pending scene set: " << filename << std::endl;
}

bool ViewportGL::loadOBJ(const std::string& filename) {
	// Clear existing meshes
	meshes.clear();
	sceneMaterials.clear();
	meshMaterialIndices.clear();
	lights.clear();

	// Create new mesh
	auto mesh = std::make_unique<Mesh>();

	if (!mesh->load(filename)) {
		std::cerr << "[Viewport] Error: Could not load mesh: " << filename << std::endl;
		return false;
	}

	if (!mesh->loadShader("assets/shaders/PhongShading.vert", "assets/shaders/PhongShading.frag")) {
		std::cerr << "[Viewport] Error: Could not load shaders" << std::endl;
		return false;
	}

	mesh->setPosition(0.0f, -0.5f, 0.0f);
	mesh->setScale(1.0f);

	// Create default material for ray tracing
	auto material = std::make_shared<graphics::Materialf>();
	material->setAmbient(Color4f(0.1f, 0.1f, 0.1f, 1.0f));
	material->setDiffuse(Color4f(0.8f, 0.6f, 0.4f, 1.0f));
	material->setSpecular(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
	material->setShininess(32.0f);
	sceneMaterials.push_back(material);
	meshMaterialIndices.push_back(0);  // OBJ mesh uses first (only) material

	meshes.push_back(std::move(mesh));

	// For OBJ files, all meshes are real geometry (no GL sphere approximations)
	numUSDMeshes = meshes.size();

	// Add default light
	Lightf light(Vector3f(3.0f, 3.0f, 3.0f), Color3f(1.0f, 1.0f, 1.0f), 1.0f);
	lights.push_back(light);

	// RT scene will be built when Start Render is clicked

	currentSceneFile = filename;
	sceneLoaded = true;

	std::cout << "[Viewport] Loaded OBJ: " << filename << std::endl;
	this->update();

	return true;
}

bool ViewportGL::loadScene(const std::string& filename) {
	// Dispatch based on file extension
	std::string ext = filename.substr(filename.find_last_of('.') + 1);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	if (ext == "usda" || ext == "usdc" || ext == "usd") {
		return loadUSD(filename);
	} else {
		return loadOBJ(filename);
	}
}

bool ViewportGL::loadUSD(const std::string& filename) {
	// Parse USDA file FIRST before building any scene data
	usd::USDAParser parser;
	usd::USDAScene usdScene;

	if (!parser.parse(filename, usdScene)) {
		std::cerr << "[Viewport] Error parsing USD: " << parser.getError() << std::endl;
		return false;
	}

	// Clear existing scene
	meshes.clear();
	sceneMaterials.clear();
	meshMaterialIndices.clear();
	meshRTMaterials.clear();
	meshNames.clear();
	spherePrimitives.clear();
	lights.clear();
	scene.reset();

	// Ensure we have GL context for creating meshes
	makeCurrent();

	// Build material lookup maps (GL for rendering, RT for ray tracing)
	std::map<std::string, std::shared_ptr<graphics::Materialf>> materialMap;
	std::map<std::string, MaterialInfo> rtMaterialMap;
	for (const auto& usdMat : usdScene.materials) {
		auto mat = std::make_shared<graphics::Materialf>();

		// Set Phong properties for OpenGL
		// Ambient should be visible even in shadow - use 0.15 factor
		mat->setAmbient(Color4f(usdMat.diffuseColor.r * 0.15f,
		                        usdMat.diffuseColor.g * 0.15f,
		                        usdMat.diffuseColor.b * 0.15f, 1.0f));
		mat->setDiffuse(Color4f(usdMat.diffuseColor.r,
		                        usdMat.diffuseColor.g,
		                        usdMat.diffuseColor.b, 1.0f));
		// Calculate shininess from roughness (roughness=1 means matte, roughness=0 means shiny)
		float shininess = (1.0f - usdMat.roughness) * 128.0f;

		// For very rough materials (low shininess), disable specular to avoid artifacts
		// When shininess approaches 0, pow(x,0)=1 causes full specular everywhere
		if (shininess < 1.0f) {
			mat->setSpecular(Color4f(0.0f, 0.0f, 0.0f, 1.0f));  // No specular for matte materials
			mat->setShininess(1.0f);  // Minimum to avoid pow(x,0) issues
		} else {
			mat->setSpecular(Color4f(usdMat.specularColor.r,
			                         usdMat.specularColor.g,
			                         usdMat.specularColor.b, 1.0f));
			mat->setShininess(shininess);
		}

		// Create RT MaterialInfo with metallic/roughness/glass properties
		MaterialInfo rtMat;
		rtMat.albedo = color(usdMat.diffuseColor.r, usdMat.diffuseColor.g, usdMat.diffuseColor.b);
		rtMat.roughness = usdMat.roughness;
		rtMat.metallic = usdMat.metallic;
		rtMat.ior = usdMat.ior;
		rtMat.is_glass = (usdMat.opacity < 0.9f);
		rtMaterialMap[usdMat.name] = rtMat;

		materialMap[usdMat.name] = mat;
		sceneMaterials.push_back(mat);

		std::cout << "[Viewport] Created material: " << usdMat.name
		          << " (metallic=" << usdMat.metallic << ", roughness=" << usdMat.roughness << ")" << std::endl;
	}

	// Create default material if none defined
	if (sceneMaterials.empty()) {
		auto defaultMat = std::make_shared<graphics::Materialf>();
		defaultMat->setAmbient(Color4f(0.1f, 0.1f, 0.1f, 1.0f));
		defaultMat->setDiffuse(Color4f(0.8f, 0.8f, 0.8f, 1.0f));
		defaultMat->setSpecular(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
		defaultMat->setShininess(32.0f);
		sceneMaterials.push_back(defaultMat);
	}

	// Build meshes from parsed data
	for (const auto& usdMesh : usdScene.meshes) {
		// Skip meshes without geometry
		if (usdMesh.points.empty() || usdMesh.faceVertexIndices.empty()) {
			continue;
		}

		// Convert USD mesh data to our Vertex/TriangleFace format
		std::vector<Vertex> vertices;
		std::vector<TriangleFace> faces;

		// Convert points to vertices
		for (size_t i = 0; i < usdMesh.points.size(); i++) {
			Vertex v;
			v.position = Vector3f(usdMesh.points[i].x,
			                      usdMesh.points[i].y,
			                      usdMesh.points[i].z);

			// Apply transform
			v.position.x() = v.position.x() * usdMesh.scale.x + usdMesh.translate.x;
			v.position.y() = v.position.y() * usdMesh.scale.y + usdMesh.translate.y;
			v.position.z() = v.position.z() * usdMesh.scale.z + usdMesh.translate.z;

			// Set normal if available
			if (i < usdMesh.normals.size()) {
				v.normal = Vector3f(usdMesh.normals[i].x,
				                    usdMesh.normals[i].y,
				                    usdMesh.normals[i].z);
			} else {
				v.normal = Vector3f(0, 1, 0);  // Default up normal
			}

			vertices.push_back(v);
		}

		// Convert faces (triangulate quads and n-gons using fan triangulation)
		size_t indexOffset = 0;
		for (size_t faceIdx = 0; faceIdx < usdMesh.faceVertexCounts.size(); faceIdx++) {
			int vertCount = usdMesh.faceVertexCounts[faceIdx];

			if (vertCount >= 3) {
				// Fan triangulation: v0, v1, v2, then v0, v2, v3, etc.
				int v0 = usdMesh.faceVertexIndices[indexOffset];
				for (int i = 1; i < vertCount - 1; i++) {
					int v1 = usdMesh.faceVertexIndices[indexOffset + i];
					int v2 = usdMesh.faceVertexIndices[indexOffset + i + 1];
					TriangleFace face;
					face.indices[0] = static_cast<unsigned int>(v0);
					face.indices[1] = static_cast<unsigned int>(v1);
					face.indices[2] = static_cast<unsigned int>(v2);
					faces.push_back(face);
				}
			}

			indexOffset += vertCount;
		}

		// Compute normals if not provided
		if (usdMesh.normals.empty()) {
			// First compute mesh centroid for orienting normals outward
			Vector3f meshCentroid(0, 0, 0);
			for (const auto& v : vertices) {
				meshCentroid = meshCentroid + v.position;
			}
			meshCentroid = meshCentroid * (1.0f / vertices.size());

			// Compute face normals and accumulate to vertices
			std::vector<Vector3f> vertexNormals(vertices.size(), Vector3f(0, 0, 0));
			for (const auto& face : faces) {
				Vector3f v0 = vertices[face.indices[0]].position;
				Vector3f v1 = vertices[face.indices[1]].position;
				Vector3f v2 = vertices[face.indices[2]].position;
				Vector3f edge1 = v1 - v0;
				Vector3f edge2 = v2 - v0;
				Vector3f faceNormal = Vector3f::Cross(edge1, edge2).normalized();

				// Compute face centroid
				Vector3f faceCentroid = (v0 + v1 + v2) * (1.0f / 3.0f);

				// Ensure normal points away from mesh centroid (outward)
				Vector3f toFace = faceCentroid - meshCentroid;
				if (Vector3f::Dot(faceNormal, toFace) < 0) {
					faceNormal = faceNormal * -1.0f;  // Flip normal to point outward
				}

				vertexNormals[face.indices[0]] = vertexNormals[face.indices[0]] + faceNormal;
				vertexNormals[face.indices[1]] = vertexNormals[face.indices[1]] + faceNormal;
				vertexNormals[face.indices[2]] = vertexNormals[face.indices[2]] + faceNormal;
			}
			for (size_t i = 0; i < vertices.size(); i++) {
				vertices[i].normal = vertexNormals[i].normalized();
			}
		}

		// Create mesh
		auto mesh = std::make_unique<Mesh>();
		mesh->setData(vertices, faces);

		if (!mesh->loadShader("assets/shaders/PhongShading.vert", "assets/shaders/PhongShading.frag")) {
			std::cerr << "[Viewport] Error loading shaders for USD mesh" << std::endl;
			continue;
		}

		// Don't apply transform again - already applied to vertices
		mesh->setPosition(0.0f, 0.0f, 0.0f);
		mesh->setScale(1.0f);

		// Look up material index from material binding
		size_t materialIndex = 0;  // Default to first material
		if (!usdMesh.materialBinding.empty() && materialMap.count(usdMesh.materialBinding)) {
			// Find the index of this material in sceneMaterials
			auto it = std::find(sceneMaterials.begin(), sceneMaterials.end(), materialMap[usdMesh.materialBinding]);
			if (it != sceneMaterials.end()) {
				materialIndex = std::distance(sceneMaterials.begin(), it);
			}
		}
		meshMaterialIndices.push_back(materialIndex);

		// Store RT material for this mesh (for CPU ray tracing with proper metallic/roughness)
		if (!usdMesh.materialBinding.empty() && rtMaterialMap.count(usdMesh.materialBinding)) {
			meshRTMaterials.push_back(rtMaterialMap[usdMesh.materialBinding]);
		} else {
			// Default Lambertian material
			MaterialInfo defaultMat;
			defaultMat.albedo = color(0.8, 0.8, 0.8);
			defaultMat.metallic = 0.0;
			defaultMat.roughness = 1.0;
			defaultMat.is_glass = false;
			meshRTMaterials.push_back(defaultMat);
		}

		meshes.push_back(std::move(mesh));
		meshNames.push_back(usdMesh.name);

		std::cout << "[Viewport] Created mesh '" << usdMesh.name << "' with material '" << usdMesh.materialBinding
		          << "' (index " << materialIndex << "): "
		          << vertices.size() << " vertices, " << faces.size() << " triangles" << std::endl;
	}

	// Remember count of actual USD mesh geometry (not GL sphere approximations)
	numUSDMeshes = meshes.size();

	// Convert sphere primitives (RTiOW-style mathematical spheres)
	for (const auto& usdSphere : usdScene.spheres) {
		SphereData sphereData;
		sphereData.center = point3(usdSphere.center.x, usdSphere.center.y, usdSphere.center.z);
		sphereData.radius = usdSphere.radius;
		sphereData.name = usdSphere.name;

		// Find USD material for sphere to extract RT properties
		const usd::USDAMaterial* usdMat = nullptr;
		for (const auto& m : usdScene.materials) {
			if (m.name == usdSphere.materialBinding) {
				usdMat = &m;
				break;
			}
		}

		// Create MaterialInfo for RT
		MaterialInfo matInfo;
		if (usdMat) {
			matInfo.albedo = color(usdMat->diffuseColor.r, usdMat->diffuseColor.g, usdMat->diffuseColor.b);
			matInfo.roughness = usdMat->roughness;
			matInfo.metallic = usdMat->metallic;
			matInfo.ior = usdMat->ior;
			matInfo.is_glass = (usdMat->opacity < 0.9f);
		} else {
			matInfo.albedo = color(0.8, 0.8, 0.8);
		}
		sphereData.material = matInfo;

		// Create GL material from RT material info for sphere rendering
		auto sphereGLMat = std::make_shared<graphics::Materialf>();
		float r = static_cast<float>(matInfo.albedo.x());
		float g = static_cast<float>(matInfo.albedo.y());
		float b = static_cast<float>(matInfo.albedo.z());

		// Handle glass/dielectric - show as light grey with high specularity
		if (matInfo.is_glass) {
			sphereGLMat->setAmbient(Color4f(0.1f, 0.1f, 0.1f, 1.0f));
			sphereGLMat->setDiffuse(Color4f(0.6f, 0.6f, 0.65f, 1.0f));  // Slight blue tint
			sphereGLMat->setSpecular(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
			sphereGLMat->setShininess(128.0f);
		} else if (matInfo.metallic > 0.5f) {
			// Metals - colored specular, moderate diffuse
			sphereGLMat->setAmbient(Color4f(r * 0.15f, g * 0.15f, b * 0.15f, 1.0f));
			sphereGLMat->setDiffuse(Color4f(r * 0.4f, g * 0.4f, b * 0.4f, 1.0f));
			sphereGLMat->setSpecular(Color4f(r, g, b, 1.0f));
			sphereGLMat->setShininess(64.0f + (1.0f - matInfo.roughness) * 64.0f);
		} else {
			// Lambertian/diffuse - full color diffuse
			sphereGLMat->setAmbient(Color4f(r * 0.15f, g * 0.15f, b * 0.15f, 1.0f));
			sphereGLMat->setDiffuse(Color4f(r, g, b, 1.0f));
			sphereGLMat->setSpecular(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
			sphereGLMat->setShininess(16.0f);
		}

		spherePrimitives.push_back(sphereData);
		std::cout << "[Viewport] Created sphere primitive '" << usdSphere.name << "' at ("
		          << sphereData.center[0] << "," << sphereData.center[1] << "," << sphereData.center[2]
		          << ") r=" << sphereData.radius << " material=" << usdSphere.materialBinding << std::endl;

		// Create GL mesh for sphere
		std::vector<Vertex> sphereVerts;
		std::vector<TriangleFace> sphereFaces;
		Vector3f glCenter(sphereData.center[0], sphereData.center[1], sphereData.center[2]);

		// For large spheres (ground), use a flat plane instead to avoid visible curvature
		double radius = sphereData.radius;
		std::cout << "[Viewport] Checking radius " << radius << " > 50.0 ? " << (radius > 50.0 ? "YES" : "NO") << std::endl;
		if (radius > 50.0) {
			// Ground plane at the top of the sphere (where objects sit)
			float groundY = glCenter.y() + static_cast<float>(radius);
			generateGroundPlane(sphereVerts, sphereFaces, groundY, 500.0f);  // Large plane
			std::cout << "[Viewport] >>> USING GROUND PLANE at y=" << groundY << std::endl;
		} else {
			generateSphereMesh(sphereVerts, sphereFaces, glCenter, static_cast<float>(radius), 32, 16);
			std::cout << "[Viewport] >>> Using sphere mesh" << std::endl;
		}

		auto sphereMesh = std::make_unique<Mesh>();
		sphereMesh->setData(sphereVerts, sphereFaces);

		if (!sphereMesh->loadShader("assets/shaders/PhongShading.vert", "assets/shaders/PhongShading.frag")) {
			std::cerr << "[Viewport] Error loading shaders for sphere mesh" << std::endl;
			continue;
		}

		sphereMesh->setPosition(0.0f, 0.0f, 0.0f);  // Position already baked into vertices
		sphereMesh->setScale(1.0f);

		meshes.push_back(std::move(sphereMesh));
		// Track material index for this sphere mesh
		size_t sphereMatIdx = sceneMaterials.size();
		sceneMaterials.push_back(sphereGLMat);
		meshMaterialIndices.push_back(sphereMatIdx);

		std::cout << "[Viewport] Created GL mesh: " << sphereVerts.size() << " vertices, "
		          << sphereFaces.size() << " triangles" << std::endl;
	}

	// Convert lights
	for (const auto& usdLight : usdScene.lights) {
		Lightf light;
		light.position = Vector3f(usdLight.position.x,
		                          usdLight.position.y,
		                          usdLight.position.z);
		light.direction = Vector3f(usdLight.direction.x,
		                           usdLight.direction.y,
		                           usdLight.direction.z);
		light.color = Color3f(usdLight.color.r,
		                      usdLight.color.g,
		                      usdLight.color.b);
		light.intensity = usdLight.intensity;
		light.radius = usdLight.radius;
		// Convert USDA light type to graphics::LightType
		switch (usdLight.type) {
			case usd::USDALightType::Point:   light.type = LightType::Point; break;
			case usd::USDALightType::Distant: light.type = LightType::Distant; break;
			case usd::USDALightType::Sphere:  light.type = LightType::Sphere; break;
			case usd::USDALightType::Rect:    light.type = LightType::Rect; break;
			default: light.type = LightType::Point; break;
		}
		lights.push_back(light);

		std::cout << "[Viewport] Created light type=" << static_cast<int>(light.type)
		          << " at (" << light.position.x() << ", "
		          << light.position.y() << ", "
		          << light.position.z() << ") intensity=" << light.intensity << std::endl;
	}

	// Add default light if none defined
	if (lights.empty()) {
		Lightf defaultLight(Vector3f(3.0f, 3.0f, 3.0f), Color3f(1.0f, 1.0f, 1.0f), 1.0f);
		lights.push_back(defaultLight);
		std::cout << "[Viewport] Added default light" << std::endl;
	}

	doneCurrent();

	// RT scene will be built when Start Render is clicked

	currentSceneFile = filename;
	sceneLoaded = true;

	std::cout << "[Viewport] Loaded USD: " << filename << std::endl;
	std::cout << "[Viewport] Total: " << meshes.size() << " meshes, "
	          << spherePrimitives.size() << " sphere primitives, "
	          << lights.size() << " lights" << std::endl;

	this->update();
	return true;
}

void ViewportGL::rebuildRayTracingScene(bool useBVH) {
	sceneBuilder.clear();

	// Add only USD meshes (not GL sphere approximations) for ray tracing
	for (size_t i = 0; i < numUSDMeshes; i++) {
		// Use stored RT material (has proper metallic/roughness/glass info)
		MaterialInfo matInfo;
		if (i < meshRTMaterials.size()) {
			matInfo = meshRTMaterials[i];
		} else {
			// Fallback: convert GL material (loses metallic/glass info)
			auto glMat = (i < sceneMaterials.size()) ? sceneMaterials[i] : sceneMaterials[0];
			Color4f diff = glMat->getDiffuse();
			matInfo.albedo = color(diff.getR(), diff.getG(), diff.getB());
			matInfo.roughness = 1.0 - (glMat->getShininess() / 128.0);
			matInfo.metallic = 0.0;
			matInfo.is_glass = false;
		}

		sceneBuilder.add_mesh(*meshes[i], matInfo);
	}

	// Add all sphere primitives (RTiOW-style mathematical spheres)
	std::cout << "[rebuildRayTracingScene] Adding " << spherePrimitives.size() << " sphere primitives..." << std::endl;
	for (size_t i = 0; i < spherePrimitives.size(); i++) {
		const auto& sphereData = spherePrimitives[i];
		std::cout << "[rebuildRayTracingScene] Sphere " << i << ": center=("
		          << sphereData.center[0] << "," << sphereData.center[1] << "," << sphereData.center[2]
		          << ") radius=" << sphereData.radius << std::endl;
		sceneBuilder.add_sphere(sphereData.center, sphereData.radius, sphereData.material);
	}

	// Add all lights
	for (const auto& light : lights) {
		sceneBuilder.add_light(light);
	}

	// Build scene (with or without BVH)
	if (useBVH) {
		scene = sceneBuilder.build_bvh();
	} else {
		scene = sceneBuilder.build_list();
	}
	rayTracer->set_scene(scene);
	rayTracer->set_lights(lights);  // Pass lights for direct illumination with shadows
	rayTracer->set_samples_per_pixel(samplesPerPixel);
	rayTracer->set_max_depth(maxDepth);

	std::cout << "[Viewport] Ray tracing scene built with " << sceneBuilder.get_object_count() << " objects ("
	          << meshes.size() << " meshes, " << spherePrimitives.size() << " spheres)" << std::endl;
}

void ViewportGL::resizeGL(int width, int height) {
	auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());
    float aspect = width / static_cast<float>(height ? height : 1);
	f->glViewport(0, 0, width, height);
	this->camera->setPerspective(camera->getFOV(), aspect, camera->getNearPlane(), camera->getFarPlane());
}

void ViewportGL::paintGL() {
	auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

	// OpenGL Phong shading mode
	{
		// Force proper GL state in case something corrupted it
		// Sky blue background to match ray tracer
		f->glClearColor(0.75f, 0.85f, 1.0f, 1.0f);
		f->glEnable(GL_DEPTH_TEST);
		f->glDepthFunc(GL_LESS);
		f->glDepthMask(GL_TRUE);
		f->glDisable(GL_BLEND);
		f->glDisable(GL_CULL_FACE);
		f->glViewport(0, 0, width(), height());

		f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render all meshes
		for (size_t i = 0; i < meshes.size(); i++) {
			const auto& mesh = meshes[i];
			if (!mesh || !mesh->getShader()) continue;

			// Get material for this mesh using the material index mapping
			size_t matIdx = (i < meshMaterialIndices.size()) ? meshMaterialIndices[i] : 0;
			auto material = (matIdx < sceneMaterials.size()) ? sceneMaterials[matIdx] : sceneMaterials[0];

			Matrix4f model = mesh->getTransform().toMatrix();
			Matrix4f view = camera->getViewMatrix();
			Matrix4f modelViewMatrix = view * model;
			Matrix4f normalMatrix = Matrix4f::Transpose(modelViewMatrix.toInverse());
			Matrix4f projectionMatrix = camera->getProjectionMatrix();

			mesh->beginRender();
			mesh->getShader()->uniformMatrix("modelViewMatrix", modelViewMatrix);
			mesh->getShader()->uniformMatrix("projectionMatrix", projectionMatrix);
			mesh->getShader()->uniformMatrix("normalMatrix", normalMatrix);
			// Find point and directional lights from scene
			Vector3f lightPos(3.0f, 3.0f, 3.0f);
			Vector3f lightDir(0.0f, -1.0f, 0.0f);
			Color3f pointLightColor(1.0f, 1.0f, 1.0f);
			Color3f dirLightColor(0.0f, 0.0f, 0.0f);
			float ambientIntensity = 0.1f;
			bool hasPointLight = false;
			bool hasDirectionalLight = false;

			for (const auto& light : lights) {
				// Scale intensity: USDA uses 0-100+, OpenGL expects ~0-1
				// Divide by 100 then clamp to reasonable range
				float scaledIntensity = std::min(light.intensity / 100.0f, 1.0f);
				// Ensure at least some light if intensity was set
				if (light.intensity > 0 && scaledIntensity < 0.1f) scaledIntensity = 0.1f;

				if (light.type == LightType::Point || light.type == LightType::Sphere) {
					lightPos = light.position;
					pointLightColor = Color3f(light.color.getR() * scaledIntensity,
					                          light.color.getG() * scaledIntensity,
					                          light.color.getB() * scaledIntensity);
					ambientIntensity = std::max(ambientIntensity, light.ambient);
					hasPointLight = true;
				} else if (light.type == LightType::Distant) {
					lightDir = light.direction;
					dirLightColor = Color3f(light.color.getR() * scaledIntensity,
					                        light.color.getG() * scaledIntensity,
					                        light.color.getB() * scaledIntensity);
					ambientIntensity = std::max(ambientIntensity, light.ambient);
					hasDirectionalLight = true;
				}
			}

			// Default to a point light if no lights in scene
			if (!hasPointLight && !hasDirectionalLight) {
				hasPointLight = true;
				pointLightColor = Color3f(0.8f, 0.8f, 0.8f);  // Slightly dimmer default
				ambientIntensity = 0.15f;
			}

			mesh->getShader()->uniformVector("lightPosition", lightPos);
			mesh->getShader()->uniformVector("lightDirection", lightDir);

			// Light intensity uniforms - use scene light colors
			mesh->getShader()->uniform4f("Ia", ambientIntensity, ambientIntensity, ambientIntensity, 1.0f);
			mesh->getShader()->uniform4f("Id", pointLightColor.getR(), pointLightColor.getG(), pointLightColor.getB(), 1.0f);
			mesh->getShader()->uniform4f("Is", 1.0f, 1.0f, 1.0f, 1.0f);  // White specular
			mesh->getShader()->uniform4f("IdDir", dirLightColor.getR(), dirLightColor.getG(), dirLightColor.getB(), 1.0f);
			mesh->getShader()->uniform1i("hasDirectionalLight", hasDirectionalLight ? 1 : 0);
			mesh->getShader()->uniform1i("hasPointLight", hasPointLight ? 1 : 0);

			// Pass material properties to shader
			Color4f ambient = material->getAmbient();
			Color4f diffuse = material->getDiffuse();
			Color4f specular = material->getSpecular();

			static bool debugOnce = true;
			if (debugOnce) {
				std::cout << "[paintGL] Material - Ambient: (" << ambient.getR() << "," << ambient.getG() << "," << ambient.getB() << ")"
				          << " Diffuse: (" << diffuse.getR() << "," << diffuse.getG() << "," << diffuse.getB() << ")"
				          << " Specular: (" << specular.getR() << "," << specular.getG() << "," << specular.getB() << ")"
				          << " Shininess: " << material->getShininess() << std::endl;
				debugOnce = false;
			}

			mesh->getShader()->uniform4f("Ka", ambient.getR(), ambient.getG(), ambient.getB(), ambient.getA());
			mesh->getShader()->uniform4f("Kd", diffuse.getR(), diffuse.getG(), diffuse.getB(), diffuse.getA());
			mesh->getShader()->uniform4f("Ks", specular.getR(), specular.getG(), specular.getB(), specular.getA());
			mesh->getShader()->uniform1f("shininess", material->getShininess());

			mesh->endRender();
		}
	}
}

void ViewportGL::mousePressEvent(QMouseEvent* e) {
	if (e->button() == Qt::RightButton) {
		isDragging = true;
		lastMousePos = e->pos();
		setCursor(Qt::ClosedHandCursor);
	}
}

void ViewportGL::mouseReleaseEvent(QMouseEvent* e) {
	if (e->button() == Qt::RightButton) {
		isDragging = false;
		setCursor(Qt::ArrowCursor);
	}
}

void ViewportGL::mouseMoveEvent(QMouseEvent* e) {
	if (!isDragging) return;

	QPoint delta = e->pos() - lastMousePos;
	lastMousePos = e->pos();

	// Rotate camera based on mouse movement
	float sensitivity = 0.005f;
	camYaw -= delta.x() * sensitivity;
	camPitch += delta.y() * sensitivity;

	// Clamp pitch to avoid flipping
	float maxPitch = 1.5f;  // ~86 degrees
	if (camPitch > maxPitch) camPitch = maxPitch;
	if (camPitch < -maxPitch) camPitch = -maxPitch;

	updateCameraFromOrbit();
	this->update();
}

void ViewportGL::wheelEvent(QWheelEvent* e) {
	// Zoom in/out
	float zoomFactor = 0.001f;
	camDistance -= e->angleDelta().y() * zoomFactor * camDistance;

	// Clamp distance
	if (camDistance < 1.0f) camDistance = 1.0f;
	if (camDistance > 100.0f) camDistance = 100.0f;

	updateCameraFromOrbit();
	this->update();
}

void ViewportGL::updateCameraFromOrbit() {
	// Calculate camera position from spherical coordinates
	float x = camDistance * cos(camPitch) * sin(camYaw);
	float y = camDistance * sin(camPitch) + camTargetY;
	float z = camDistance * cos(camPitch) * cos(camYaw);

	Vector3f eye(x, y, z);
	Vector3f lookAt(0.0f, camTargetY, 0.0f);
	Vector3f up(0.0f, 1.0f, 0.0f);

	// Update camera properties
	camera->getEye() = eye;
	camera->getLookAt() = lookAt;
	camera->getUp() = up;

	// Compute view matrix directly using LookAt
	camera->getViewMatrix() = Matrix4f::LookAt(eye, lookAt, up);
}

void ViewportGL::keyPressEvent(QKeyEvent* e) {
	std::cout << "[Viewport] Key pressed: " << e->key() << std::endl;
	QOpenGLWidget::keyPressEvent(e);
}

void ViewportGL::setSamplesPerPixel(int samples) {
	samplesPerPixel = std::max(1, samples);
	if (rayTracer) {
		rayTracer->set_samples_per_pixel(samplesPerPixel);
	}
	std::cout << "[Viewport] Samples per pixel: " << samplesPerPixel << std::endl;
	this->update();
}

void ViewportGL::setMaxDepth(int depth) {
	maxDepth = std::max(1, depth);
	if (rayTracer) {
		rayTracer->set_max_depth(maxDepth);
	}
	std::cout << "[Viewport] Max depth: " << maxDepth << std::endl;
	this->update();
}

void ViewportGL::startBatchRender(const RenderSettings& settings) {
	if (batchRendering) {
		std::cerr << "[Viewport] Already rendering!" << std::endl;
		return;
	}

	if (!sceneLoaded) {
		emit batchRenderComplete(false, "No scene loaded");
		return;
	}

	batchRendering = true;
	rayTracer->reset_cancel();

	// Build RT scene with BVH setting (only done here, not at load time)
	std::cout << "[Viewport] Rebuilding scene with BVH=" << (settings.use_bvh ? "true" : "false") << std::endl;
	rebuildRayTracingScene(settings.use_bvh);

	// Stop GL render loop to maximize CPU for ray tracing
	timer->stop();
	std::cout << "[Viewport] Paused GL render loop for batch rendering" << std::endl;

	std::cout << "[Viewport] Starting batch render..." << std::endl;
	std::cout << "  Resolution: " << settings.width << "x" << settings.height << std::endl;
	std::cout << "  Frames: " << settings.start_frame << " to " << settings.end_frame << std::endl;
	std::cout << "  Samples: " << settings.samples_per_pixel << ", Depth: " << settings.max_depth << std::endl;
	std::cout << "  BVH: " << (settings.use_bvh ? "enabled" : "disabled") << std::endl;
	std::cout << "  Output: " << settings.output_directory.toStdString() << std::endl;

	// Capture camera state for render thread
	Cameraf cameraCopy = *camera;

	// Create worker thread
	renderThread = QThread::create([this, settings, cameraCopy]() {
		// Progress callback - emits signal (thread-safe via Qt queued connection)
		auto progress = [this](int frame, int total) {
			QMetaObject::invokeMethod(this, [=]() {
				emit batchRenderProgress(frame, total);
			}, Qt::QueuedConnection);
		};

		// Frame complete callback with timing
		auto frameComplete = [this](double frameTime) {
			QMetaObject::invokeMethod(this, [=]() {
				emit batchRenderFrameComplete(frameTime);
			}, Qt::QueuedConnection);
		};

		// Completion callback
		auto completion = [this](bool success, const QString& msg) {
			QMetaObject::invokeMethod(this, [=]() {
				batchRendering = false;
				// Resume GL render loop
				timer->start();
				std::cout << "[Viewport] Resumed GL render loop" << std::endl;
				emit batchRenderComplete(success, msg);
			}, Qt::QueuedConnection);
		};

		// Run the batch render (blocks until done or cancelled)
		rayTracer->render_animation(settings, cameraCopy, progress, frameComplete, completion);
	});

	renderThread->start();
}

void ViewportGL::cancelBatchRender() {
	if (rayTracer && batchRendering) {
		std::cout << "[Viewport] Cancelling batch render..." << std::endl;
		rayTracer->request_cancel();
	}
}

// ============== Light/Material Modification ==============

int ViewportGL::getLightCount() const {
    return static_cast<int>(lights.size());
}

QStringList ViewportGL::getObjectNames() const {
    QStringList names;
    // Add spheres first
    for (size_t i = 0; i < spherePrimitives.size(); ++i) {
        if (!spherePrimitives[i].name.empty()) {
            names.append(QString::fromStdString(spherePrimitives[i].name));
        } else {
            names.append(QString("Sphere %1").arg(i));
        }
    }
    // Add meshes
    for (size_t i = 0; i < numUSDMeshes && i < meshNames.size(); ++i) {
        if (!meshNames[i].empty()) {
            names.append(QString::fromStdString(meshNames[i]));
        } else {
            names.append(QString("Mesh %1").arg(i));
        }
    }
    return names;
}

void ViewportGL::getLightValues(int index, float& r, float& g, float& b, float& intensity) const {
    if (index >= 0 && index < static_cast<int>(lights.size())) {
        r = lights[index].color.r();
        g = lights[index].color.g();
        b = lights[index].color.b();
        intensity = lights[index].intensity;
    }
}

void ViewportGL::getMaterialValues(int index, int& type, float& r, float& g, float& b, float& param) const {
    int numSpheres = static_cast<int>(spherePrimitives.size());

    if (index >= 0 && index < numSpheres) {
        // It's a sphere
        const auto& mat = spherePrimitives[index].material;
        // Convert roughness/metallic to type: 0=lambertian, 1=metal, 2=glass
        if (mat.is_glass) {
            type = 2;  // Glass/Dielectric
            param = static_cast<float>(mat.ior);
        } else if (mat.metallic > 0.5) {
            type = 1;  // Metal
            param = static_cast<float>(mat.roughness);  // roughness as fuzz
        } else {
            type = 0;  // Lambertian
            param = 0.0f;
        }
        r = static_cast<float>(mat.albedo.x());
        g = static_cast<float>(mat.albedo.y());
        b = static_cast<float>(mat.albedo.z());
    } else {
        // It's a mesh
        int meshIndex = index - numSpheres;
        if (meshIndex >= 0 && meshIndex < static_cast<int>(meshRTMaterials.size())) {
            const auto& mat = meshRTMaterials[meshIndex];
            if (mat.is_glass) {
                type = 2;
                param = static_cast<float>(mat.ior);
            } else if (mat.metallic > 0.5) {
                type = 1;
                param = static_cast<float>(mat.roughness);
            } else {
                type = 0;
                param = 0.0f;
            }
            r = static_cast<float>(mat.albedo.x());
            g = static_cast<float>(mat.albedo.y());
            b = static_cast<float>(mat.albedo.z());
        }
    }
}

void ViewportGL::updateLight(int index, float r, float g, float b, float intensity) {
    if (index >= 0 && index < static_cast<int>(lights.size())) {
        lights[index].color = Color3f(r, g, b);
        lights[index].intensity = intensity;

        // Update ray tracer lights
        if (rayTracer) {
            rayTracer->set_lights(lights);
        }
    }
    update();
}

void ViewportGL::updateMaterial(int objectIndex, int materialType, float r, float g, float b, float param) {
    int numSpheres = static_cast<int>(spherePrimitives.size());
    int numMeshes = static_cast<int>(meshRTMaterials.size());

    if (objectIndex < 0 || objectIndex >= numSpheres + numMeshes) {
        return;
    }

    // Helper to update OpenGL material for Phong shading
    auto updateGLMaterial = [&](size_t glMatIndex) {
        if (glMatIndex >= sceneMaterials.size()) return;
        auto& glMat = sceneMaterials[glMatIndex];

        // Handle glass/dielectric - show as light grey with high specularity
        if (materialType == 2) { // Glass
            glMat->setAmbient(Color4f(0.1f, 0.1f, 0.1f, 1.0f));
            glMat->setDiffuse(Color4f(0.6f, 0.6f, 0.65f, 1.0f));  // Slight blue tint
            glMat->setSpecular(Color4f(1.0f, 1.0f, 1.0f, 1.0f));
            glMat->setShininess(128.0f);
        } else if (materialType == 1) { // Metal
            glMat->setAmbient(Color4f(r * 0.15f, g * 0.15f, b * 0.15f, 1.0f));
            glMat->setDiffuse(Color4f(r * 0.4f, g * 0.4f, b * 0.4f, 1.0f));
            glMat->setSpecular(Color4f(r, g, b, 1.0f));
            glMat->setShininess(64.0f + (1.0f - param) * 64.0f);  // param is fuzz/roughness
        } else { // Lambertian/diffuse
            glMat->setAmbient(Color4f(r * 0.15f, g * 0.15f, b * 0.15f, 1.0f));
            glMat->setDiffuse(Color4f(r, g, b, 1.0f));
            glMat->setSpecular(Color4f(0.2f, 0.2f, 0.2f, 1.0f));
            glMat->setShininess(16.0f);
        }
    };

    if (objectIndex < numSpheres) {
        // Update sphere primitive material (for CPU ray tracing)
        MaterialInfo& mat = spherePrimitives[objectIndex].material;
        mat.albedo = color(r, g, b);
        if (materialType == 0) { // Lambertian
            mat.is_glass = false;
            mat.metallic = 0.0;
            mat.roughness = 1.0;
        } else if (materialType == 1) { // Metal
            mat.is_glass = false;
            mat.metallic = 1.0;
            mat.roughness = param;  // param is fuzz
        } else if (materialType == 2) { // Glass
            mat.is_glass = true;
            mat.metallic = 0.0;
            mat.ior = param;
        }

        // Update OpenGL material for sphere mesh (sphere meshes are added after USD meshes)
        // Sphere GL meshes start at index numUSDMeshes, and each sphere has one GL mesh
        size_t sphereGLMeshIndex = numUSDMeshes + objectIndex;
        if (sphereGLMeshIndex < meshMaterialIndices.size()) {
            size_t glMatIndex = meshMaterialIndices[sphereGLMeshIndex];
            updateGLMaterial(glMatIndex);
        }
    } else {
        // Update mesh material (for CPU ray tracing)
        int meshIndex = objectIndex - numSpheres;
        if (meshIndex >= 0 && meshIndex < static_cast<int>(meshRTMaterials.size())) {
            MaterialInfo& mat = meshRTMaterials[meshIndex];
            mat.albedo = color(r, g, b);
            if (materialType == 0) { // Lambertian
                mat.is_glass = false;
                mat.metallic = 0.0;
                mat.roughness = 1.0;
            } else if (materialType == 1) { // Metal
                mat.is_glass = false;
                mat.metallic = 1.0;
                mat.roughness = param;
            } else if (materialType == 2) { // Glass
                mat.is_glass = true;
                mat.metallic = 0.0;
                mat.ior = param;
            }

            // Update OpenGL material for mesh
            if (meshIndex < static_cast<int>(meshMaterialIndices.size())) {
                size_t glMatIndex = meshMaterialIndices[meshIndex];
                updateGLMaterial(glMatIndex);
            }
        }
    }

    // Rebuild ray tracing scene
    rebuildRayTracingScene(true);

    update();
}

}
