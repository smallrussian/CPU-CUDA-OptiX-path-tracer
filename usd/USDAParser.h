/*
 * Copyright (c) Saint Louis University (SLU)
 * Graphics and eXtended Reality (GXR) Laboratory
 */
#ifndef USDA_PARSER_H
#define USDA_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace usd {

//=============================================================================
// RAW DATA STRUCTURES
//=============================================================================

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Vec2 {
    float u, v;
    Vec2() : u(0), v(0) {}
    Vec2(float u_, float v_) : u(u_), v(v_) {}
};

struct Color3 {
    float r, g, b;
    Color3() : r(0.8f), g(0.8f), b(0.8f) {}
    Color3(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
};

struct USDAMesh {
    std::string name;
    std::string path;                  // Full prim path
    std::vector<Vec3> points;
    std::vector<int> faceVertexCounts;
    std::vector<int> faceVertexIndices;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    Color3 displayColor;
    bool hasDisplayColor;
    std::string materialBinding;

    // Transform (accumulated from hierarchy)
    Vec3 translate;
    Vec3 rotate;
    Vec3 scale;

    USDAMesh() : hasDisplayColor(false), scale(1,1,1) {}
};

struct USDAMaterial {
    std::string name;
    Color3 diffuseColor;
    Color3 specularColor;
    float roughness;
    float metallic;
    float ior;
    float opacity;

    USDAMaterial()
        : diffuseColor(0.8f, 0.8f, 0.8f),
          specularColor(1.0f, 1.0f, 1.0f),
          roughness(0.5f), metallic(0.0f), ior(1.5f), opacity(1.0f) {}
};

enum class USDALightType { Point, Distant, Sphere, Rect, Spot };

struct USDALight {
    std::string name;
    USDALightType type;
    Vec3 position;
    Color3 color;
    float intensity;
    float radius;
    Vec3 direction;
    float coneAngle;    // For spotlights (in degrees)
    float coneFalloff;  // Softness of spotlight edge

    USDALight()
        : type(USDALightType::Point),
          color(1.0f, 1.0f, 1.0f),
          intensity(1.0f), radius(1.0f),
          direction(0, -1, 0),
          coneAngle(45.0f),
          coneFalloff(5.0f) {}
};

// Sphere primitive (actual mathematical sphere, not mesh)
struct USDASphere {
    std::string name;
    std::string path;
    Vec3 center;          // Position (from transform)
    float radius;
    std::string materialBinding;

    USDASphere() : radius(1.0f) {}
};

struct USDAScene {
    std::string defaultPrim;
    std::string upAxis;
    std::vector<USDAMesh> meshes;
    std::vector<USDASphere> spheres;  // Actual sphere primitives
    std::vector<USDAMaterial> materials;
    std::vector<USDALight> lights;
    bool disableSky;  // If true, use black background instead of sky gradient

    USDAScene() : upAxis("Y"), disableSky(false) {}

    void clear() {
        defaultPrim.clear();
        upAxis = "Y";
        meshes.clear();
        spheres.clear();
        materials.clear();
        lights.clear();
        disableSky = false;
    }
};

//=============================================================================
// PARSER CLASS
//=============================================================================

class USDAParser {
public:
    USDAParser() = default;

    bool parse(const std::string& filename, USDAScene& outScene) {
        outScene.clear();
        m_error.clear();
        m_currentPath.clear();
        m_currentTranslate = Vec3();
        m_currentRotate = Vec3();
        m_currentScale = Vec3(1, 1, 1);

        // Extract base directory for relative paths (OBJ payloads)
        size_t lastSlash = filename.rfind('/');
        size_t lastBackslash = filename.rfind('\\');
        size_t pos = std::string::npos;
        if (lastSlash != std::string::npos && lastBackslash != std::string::npos) {
            pos = std::max(lastSlash, lastBackslash);
        } else if (lastSlash != std::string::npos) {
            pos = lastSlash;
        } else if (lastBackslash != std::string::npos) {
            pos = lastBackslash;
        }
        m_baseDirectory = (pos != std::string::npos) ? filename.substr(0, pos + 1) : "";

        std::ifstream file(filename);
        if (!file.is_open()) {
            m_error = "Could not open file: " + filename;
            return false;
        }

        std::cout << "[USDAParser] Parsing: " << filename << std::endl;
        std::cout << "[USDAParser] Base directory: " << m_baseDirectory << std::endl;

        if (!parseHeader(file, outScene)) {
            return false;
        }

        parseContent(file, outScene);

        file.close();

        std::cout << "[USDAParser] Parsed: "
                  << outScene.meshes.size() << " meshes, "
                  << outScene.spheres.size() << " sphere primitives, "
                  << outScene.materials.size() << " materials, "
                  << outScene.lights.size() << " lights" << std::endl;

        return true;
    }

    const std::string& getError() const { return m_error; }

private:
    std::string m_error;
    std::string m_currentPath;
    std::string m_baseDirectory;  // Directory of the USDA file for relative paths
    Vec3 m_currentTranslate;
    Vec3 m_currentRotate;
    Vec3 m_currentScale;

    //=========================================================================
    // HEADER PARSING
    //=========================================================================

    bool parseHeader(std::ifstream& file, USDAScene& scene) {
        std::string line;

        // Find #usda header
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty()) continue;

            if (startsWith(line, "#usda")) {
                break;
            } else {
                m_error = "Invalid USDA file: missing #usda header";
                return false;
            }
        }

        // Parse metadata block
        bool inCustomLayerData = false;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty()) continue;
            if (line[0] == '(') continue;  // Start of metadata
            if (line[0] == ')') break;     // End of metadata

            if (startsWith(line, "defaultPrim")) {
                scene.defaultPrim = extractQuotedString(line);
            } else if (startsWith(line, "upAxis")) {
                scene.upAxis = extractQuotedString(line);
            } else if (startsWith(line, "customLayerData")) {
                inCustomLayerData = true;
            } else if (inCustomLayerData) {
                if (line.find('}') != std::string::npos) {
                    inCustomLayerData = false;
                }
                // Parse disableSky flag
                if (line.find("disableSky") != std::string::npos) {
                    if (line.find("true") != std::string::npos) {
                        scene.disableSky = true;
                        std::cout << "[USDAParser] disableSky = true" << std::endl;
                    }
                }
            }
        }

        return true;
    }

    //=========================================================================
    // CONTENT PARSING
    //=========================================================================

    void parseContent(std::ifstream& file, USDAScene& scene) {
        std::string line;

        std::cout << "[USDAParser] Starting parseContent..." << std::endl;

        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            // Check for prim definitions
            if (startsWith(line, "def Mesh")) {
                std::cout << "[USDAParser] parseContent found: " << line << std::endl;
                std::string name = extractPrimName(line);
                m_currentPath += "/" + name;
                parseMesh(file, scene, name);
                // Pop path
                size_t lastSlash = m_currentPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    m_currentPath = m_currentPath.substr(0, lastSlash);
                }
            }
            else if (startsWith(line, "def Sphere") && !startsWith(line, "def SphereLight")) {
                std::cout << "[USDAParser] parseContent found Sphere: " << line << std::endl;
                std::string name = extractPrimName(line);
                m_currentPath += "/" + name;
                parseSphere(file, scene, name);
                // Pop path
                size_t lastSlash = m_currentPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    m_currentPath = m_currentPath.substr(0, lastSlash);
                }
            }
            else if (startsWith(line, "def Cube")) {
                std::cout << "[USDAParser] parseContent found Cube: " << line << std::endl;
                std::string name = extractPrimName(line);
                m_currentPath += "/" + name;
                parseCube(file, scene, name);
                // Pop path
                size_t lastSlash = m_currentPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    m_currentPath = m_currentPath.substr(0, lastSlash);
                }
            }
            else if (startsWith(line, "def Xform")) {
                std::cout << "[USDAParser] parseContent found: " << line << std::endl;
                std::string name = extractPrimName(line);
                m_currentPath += "/" + name;
                parseXform(file, scene);
                // Pop path
                size_t lastSlash = m_currentPath.rfind('/');
                if (lastSlash != std::string::npos) {
                    m_currentPath = m_currentPath.substr(0, lastSlash);
                }
            }
            else if (startsWith(line, "def Material")) {
                std::string name = extractPrimName(line);
                parseMaterial(file, scene, name);
            }
            else if (startsWith(line, "def SphereLight")) {
                std::string name = extractPrimName(line);
                parseLight(file, scene, name, USDALightType::Sphere);
            }
            else if (startsWith(line, "def DistantLight")) {
                std::string name = extractPrimName(line);
                parseLight(file, scene, name, USDALightType::Distant);
            }
            else if (startsWith(line, "def RectLight")) {
                std::string name = extractPrimName(line);
                parseLight(file, scene, name, USDALightType::Rect);
            }
            else if (startsWith(line, "def SpotLight")) {
                std::string name = extractPrimName(line);
                parseLight(file, scene, name, USDALightType::Spot);
            }
        }
    }

    //=========================================================================
    // MESH PARSING
    //=========================================================================

    void parseMesh(std::ifstream& file, USDAScene& scene, const std::string& name) {
        std::cout << "[USDAParser] >>> Entering parseMesh for '" << name << "'" << std::endl;

        USDAMesh mesh;
        mesh.name = name;
        mesh.path = m_currentPath;
        mesh.translate = m_currentTranslate;
        mesh.rotate = m_currentRotate;
        mesh.scale = m_currentScale;

        std::string line;
        int braceDepth = 0;
        std::string multiLineBuffer;
        bool inMultiLine = false;
        std::string multiLineKey;

        // Find opening brace first
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{') {
                braceDepth = 1;
                std::cout << "[USDAParser] Found opening brace for Mesh '" << name << "'" << std::endl;
                break;
            }
            // Check if brace is on same line as def
            if (trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                std::cout << "[USDAParser] Found opening brace (inline) for Mesh '" << name << "'" << std::endl;
                break;
            }
        }

        if (braceDepth == 0) {
            std::cerr << "[USDAParser] Warning: No opening brace for Mesh " << name << std::endl;
            return;
        }

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            // Track brace depth
            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // Handle multi-line arrays
            if (inMultiLine) {
                // Strip comments before adding to buffer
                std::string noComment = trim(stripComment(trimmed));
                if (!noComment.empty()) {
                    multiLineBuffer += " " + noComment;
                }
                if (trimmed.find(']') != std::string::npos) {
                    inMultiLine = false;
                    std::cout << "[USDAParser] Processing multi-line array key='" << multiLineKey << "'" << std::endl;
                    processArrayAttribute(mesh, multiLineKey, multiLineBuffer);
                    multiLineBuffer.clear();
                }
                continue;
            }

            // Check for array start - must look AFTER the '=' sign
            // because type declarations like 'float3[]' also contain brackets
            size_t eqPos = trimmed.find('=');
            if (eqPos != std::string::npos) {
                std::string afterEquals = trimmed.substr(eqPos + 1);
                if (afterEquals.find('[') != std::string::npos &&
                    afterEquals.find(']') == std::string::npos) {
                    // Multi-line array - the '[' is after '=' but ']' is not
                    multiLineKey = extractAttributeKey(trimmed);
                    multiLineBuffer = trimmed;
                    inMultiLine = true;
                    std::cout << "[USDAParser] Starting multi-line array: key='" << multiLineKey << "'" << std::endl;
                    continue;
                }
            }

            // Single-line attributes
            if (startsWith(trimmed, "point3f[] points") ||
                startsWith(trimmed, "float3[] points")) {
                mesh.points = parseVec3Array(trimmed);
            }
            else if (startsWith(trimmed, "int[] faceVertexCounts")) {
                mesh.faceVertexCounts = parseIntArray(trimmed);
            }
            else if (startsWith(trimmed, "int[] faceVertexIndices")) {
                mesh.faceVertexIndices = parseIntArray(trimmed);
            }
            else if (startsWith(trimmed, "normal3f[] normals")) {
                mesh.normals = parseVec3Array(trimmed);
            }
            else if (startsWith(trimmed, "texCoord2f[] primvars:st") ||
                     startsWith(trimmed, "float2[] primvars:st")) {
                mesh.uvs = parseVec2Array(trimmed);
            }
            else if (startsWith(trimmed, "color3f[] primvars:displayColor")) {
                auto colors = parseVec3Array(trimmed);
                if (!colors.empty()) {
                    mesh.displayColor = Color3(colors[0].x, colors[0].y, colors[0].z);
                    mesh.hasDisplayColor = true;
                }
            }
            else if (trimmed.find("rel material:binding") != std::string::npos) {
                mesh.materialBinding = extractMaterialPath(trimmed);
            }
            // Transform ops on mesh itself
            else if (startsWith(trimmed, "float3 xformOp:translate") ||
                     startsWith(trimmed, "double3 xformOp:translate")) {
                mesh.translate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:rotateXYZ") ||
                     startsWith(trimmed, "double3 xformOp:rotateXYZ")) {
                mesh.rotate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateX") ||
                     startsWith(trimmed, "double xformOp:rotateX")) {
                mesh.rotate.x = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateY") ||
                     startsWith(trimmed, "double xformOp:rotateY")) {
                mesh.rotate.y = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateZ") ||
                     startsWith(trimmed, "double xformOp:rotateZ")) {
                mesh.rotate.z = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:scale") ||
                     startsWith(trimmed, "double3 xformOp:scale")) {
                mesh.scale = parseVec3Value(trimmed);
            }
            // Handle nested mesh definitions (child meshes)
            else if (startsWith(trimmed, "def Mesh")) {
                std::string childName = extractPrimName(trimmed);
                std::string savedPath = m_currentPath;
                m_currentPath += "/" + childName;
                // Save parent transform
                Vec3 savedT = m_currentTranslate, savedR = m_currentRotate, savedS = m_currentScale;
                m_currentTranslate = mesh.translate;
                m_currentRotate = mesh.rotate;
                m_currentScale = mesh.scale;
                parseMesh(file, scene, childName);
                // Restore
                m_currentTranslate = savedT;
                m_currentRotate = savedR;
                m_currentScale = savedS;
                m_currentPath = savedPath;
                // Note: parseMesh consumes its own braces, don't decrement here
            }
        }

        // Debug output before adding mesh
        std::cout << "[USDAParser] <<< Exiting parseMesh for '" << name << "'" << std::endl;
        std::cout << "[USDAParser]     points.size() = " << mesh.points.size() << std::endl;
        std::cout << "[USDAParser]     faceVertexCounts.size() = " << mesh.faceVertexCounts.size() << std::endl;
        std::cout << "[USDAParser]     faceVertexIndices.size() = " << mesh.faceVertexIndices.size() << std::endl;

        // Print first few points for verification
        if (!mesh.points.empty()) {
            std::cout << "[USDAParser]     First 3 points: ";
            for (size_t i = 0; i < std::min((size_t)3, mesh.points.size()); i++) {
                std::cout << "(" << mesh.points[i].x << "," << mesh.points[i].y << "," << mesh.points[i].z << ") ";
            }
            std::cout << std::endl;
        }
        // Print faceVertexCounts
        if (!mesh.faceVertexCounts.empty()) {
            std::cout << "[USDAParser]     faceVertexCounts: [";
            for (size_t i = 0; i < mesh.faceVertexCounts.size(); i++) {
                std::cout << mesh.faceVertexCounts[i];
                if (i < mesh.faceVertexCounts.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        // Print first few indices
        if (!mesh.faceVertexIndices.empty()) {
            std::cout << "[USDAParser]     First 12 indices: [";
            for (size_t i = 0; i < std::min((size_t)12, mesh.faceVertexIndices.size()); i++) {
                std::cout << mesh.faceVertexIndices[i];
                if (i < 11 && i < mesh.faceVertexIndices.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // Only add mesh if it has geometry
        if (!mesh.points.empty() && !mesh.faceVertexIndices.empty()) {
            scene.meshes.push_back(mesh);
            std::cout << "[USDAParser] Mesh '" << name << "': "
                      << mesh.points.size() << " points, "
                      << mesh.faceVertexCounts.size() << " faces" << std::endl;
        } else {
            std::cout << "[USDAParser] WARNING: Mesh '" << name << "' has no geometry, NOT added!" << std::endl;
        }
    }

    void processArrayAttribute(USDAMesh& mesh, const std::string& key, const std::string& data) {
        if (key.find("points") != std::string::npos) {
            mesh.points = parseVec3Array(data);
            std::cout << "[USDAParser]   -> Parsed points: " << mesh.points.size() << " items" << std::endl;
        }
        else if (key.find("faceVertexCounts") != std::string::npos) {
            mesh.faceVertexCounts = parseIntArray(data);
            std::cout << "[USDAParser]   -> Parsed faceVertexCounts: " << mesh.faceVertexCounts.size() << " items" << std::endl;
        }
        else if (key.find("faceVertexIndices") != std::string::npos) {
            mesh.faceVertexIndices = parseIntArray(data);
            std::cout << "[USDAParser]   -> Parsed faceVertexIndices: " << mesh.faceVertexIndices.size() << " items" << std::endl;
        }
        else if (key.find("normals") != std::string::npos) {
            mesh.normals = parseVec3Array(data);
            std::cout << "[USDAParser]   -> Parsed normals: " << mesh.normals.size() << " items" << std::endl;
        }
        else if (key.find("primvars:st") != std::string::npos) {
            mesh.uvs = parseVec2Array(data);
            std::cout << "[USDAParser]   -> Parsed UVs: " << mesh.uvs.size() << " items" << std::endl;
        }
        else if (key.find("displayColor") != std::string::npos) {
            auto colors = parseVec3Array(data);
            if (!colors.empty()) {
                mesh.displayColor = Color3(colors[0].x, colors[0].y, colors[0].z);
                mesh.hasDisplayColor = true;
                std::cout << "[USDAParser]   -> Parsed displayColor" << std::endl;
            }
        }
        else {
            std::cout << "[USDAParser]   -> Unknown array key: '" << key << "'" << std::endl;
        }
    }

    //=========================================================================
    // XFORM PARSING
    //=========================================================================

    void parseXform(std::ifstream& file, USDAScene& scene) {
        std::cout << "[USDAParser] >>> Entering parseXform, path=" << m_currentPath << std::endl;

        std::string line;
        int braceDepth = 0;
        std::string payloadPath;  // For OBJ payload support
        std::string xformMaterialBinding;  // Material binding on Xform (for OBJ payloads)

        // Find opening brace, but also check for payload and material binding in prim metadata
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;

            // Check for payload = @file.obj@ syntax
            if (trimmed.find("payload") != std::string::npos && trimmed.find('@') != std::string::npos) {
                payloadPath = extractPayloadPath(trimmed);
                std::cout << "[USDAParser] Found payload: " << payloadPath << std::endl;
            }

            // Check for material binding in Xform metadata (before the {)
            if (trimmed.find("material:binding") != std::string::npos) {
                xformMaterialBinding = extractMaterialBinding(trimmed);
                std::cout << "[USDAParser] Found Xform material binding: " << xformMaterialBinding << std::endl;
            }

            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                std::cout << "[USDAParser] Found opening brace for Xform" << std::endl;
                break;
            }
        }

        if (braceDepth == 0) {
            std::cerr << "[USDAParser] Warning: No opening brace for Xform" << std::endl;
            return;
        }

        // Save parent transform
        Vec3 parentT = m_currentTranslate;
        Vec3 parentR = m_currentRotate;
        Vec3 parentS = m_currentScale;

        Vec3 localTranslate, localRotate;
        Vec3 localScale(1, 1, 1);

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // Parse transform operations
            if (startsWith(trimmed, "float3 xformOp:translate") ||
                startsWith(trimmed, "double3 xformOp:translate")) {
                localTranslate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:rotateXYZ") ||
                     startsWith(trimmed, "double3 xformOp:rotateXYZ")) {
                localRotate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateX")) {
                localRotate.x = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateY")) {
                localRotate.y = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float xformOp:rotateZ")) {
                localRotate.z = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:scale") ||
                     startsWith(trimmed, "double3 xformOp:scale")) {
                localScale = parseVec3Value(trimmed);
            }
            // Nested definitions
            else if (startsWith(trimmed, "def Mesh")) {
                std::cout << "[USDAParser] Found 'def Mesh' in Xform: " << trimmed << std::endl;

                // Apply accumulated transform
                m_currentTranslate = Vec3(
                    parentT.x + localTranslate.x,
                    parentT.y + localTranslate.y,
                    parentT.z + localTranslate.z
                );
                m_currentRotate = Vec3(
                    parentR.x + localRotate.x,
                    parentR.y + localRotate.y,
                    parentR.z + localRotate.z
                );
                m_currentScale = Vec3(
                    parentS.x * localScale.x,
                    parentS.y * localScale.y,
                    parentS.z * localScale.z
                );

                std::string meshName = extractPrimName(trimmed);
                std::string savedPath = m_currentPath;
                m_currentPath += "/" + meshName;
                parseMesh(file, scene, meshName);
                m_currentPath = savedPath;
                // Note: parseMesh consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def Sphere") && !startsWith(trimmed, "def SphereLight")) {
                // Sphere primitive inside Xform
                std::cout << "[USDAParser] Found 'def Sphere' in Xform: " << trimmed << std::endl;
                m_currentTranslate = Vec3(
                    parentT.x + localTranslate.x,
                    parentT.y + localTranslate.y,
                    parentT.z + localTranslate.z
                );
                m_currentScale = Vec3(
                    parentS.x * localScale.x,
                    parentS.y * localScale.y,
                    parentS.z * localScale.z
                );

                std::string sphereName = extractPrimName(trimmed);
                std::string savedPath = m_currentPath;
                m_currentPath += "/" + sphereName;
                parseSphere(file, scene, sphereName);
                m_currentPath = savedPath;
            }
            else if (startsWith(trimmed, "def Cube")) {
                // Cube primitive inside Xform
                std::cout << "[USDAParser] Found 'def Cube' in Xform: " << trimmed << std::endl;
                m_currentTranslate = Vec3(
                    parentT.x + localTranslate.x,
                    parentT.y + localTranslate.y,
                    parentT.z + localTranslate.z
                );
                m_currentScale = Vec3(
                    parentS.x * localScale.x,
                    parentS.y * localScale.y,
                    parentS.z * localScale.z
                );

                std::string cubeName = extractPrimName(trimmed);
                std::string savedPath = m_currentPath;
                m_currentPath += "/" + cubeName;
                parseCube(file, scene, cubeName);
                m_currentPath = savedPath;
            }
            else if (startsWith(trimmed, "def Xform")) {
                // Nested Xform
                m_currentTranslate = Vec3(
                    parentT.x + localTranslate.x,
                    parentT.y + localTranslate.y,
                    parentT.z + localTranslate.z
                );
                m_currentRotate = Vec3(
                    parentR.x + localRotate.x,
                    parentR.y + localRotate.y,
                    parentR.z + localRotate.z
                );
                m_currentScale = Vec3(
                    parentS.x * localScale.x,
                    parentS.y * localScale.y,
                    parentS.z * localScale.z
                );

                std::string xformName = extractPrimName(trimmed);
                std::string savedPath = m_currentPath;
                m_currentPath += "/" + xformName;
                parseXform(file, scene);
                m_currentPath = savedPath;
                // Note: parseXform consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def SphereLight")) {
                std::string lightName = extractPrimName(trimmed);
                parseLight(file, scene, lightName, USDALightType::Sphere);
                // Note: parseLight consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def DistantLight")) {
                std::string lightName = extractPrimName(trimmed);
                parseLight(file, scene, lightName, USDALightType::Distant);
                // Note: parseLight consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def SpotLight")) {
                std::string lightName = extractPrimName(trimmed);
                parseLight(file, scene, lightName, USDALightType::Spot);
                // Note: parseLight consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def Scope")) {
                // Scope is like Xform but without transforms - just a container
                parseScope(file, scene);
                // Note: parseScope consumes its own braces, don't decrement here
            }
            else if (startsWith(trimmed, "def Material")) {
                std::string matName = extractPrimName(trimmed);
                parseMaterial(file, scene, matName);
                // Note: parseMaterial consumes its own braces, don't decrement here
            }
        }

        // If we found a payload, load the OBJ file now that we have the transform
        if (!payloadPath.empty()) {
            // Check if it's an OBJ file
            std::string ext = payloadPath.substr(payloadPath.find_last_of('.') + 1);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == "obj") {
                std::string fullPath = m_baseDirectory + payloadPath;
                std::cout << "[USDAParser] Loading OBJ payload from: " << fullPath << std::endl;

                USDAMesh mesh;
                mesh.name = m_currentPath.substr(m_currentPath.rfind('/') + 1);
                mesh.path = m_currentPath;

                // Apply material binding from Xform metadata
                mesh.materialBinding = xformMaterialBinding;

                // Apply accumulated transform
                mesh.translate = Vec3(
                    parentT.x + localTranslate.x,
                    parentT.y + localTranslate.y,
                    parentT.z + localTranslate.z
                );
                mesh.rotate = Vec3(
                    parentR.x + localRotate.x,
                    parentR.y + localRotate.y,
                    parentR.z + localRotate.z
                );
                mesh.scale = Vec3(
                    parentS.x * localScale.x,
                    parentS.y * localScale.y,
                    parentS.z * localScale.z
                );

                if (loadOBJ(fullPath, mesh)) {
                    scene.meshes.push_back(mesh);
                    std::cout << "[USDAParser] Added OBJ mesh '" << mesh.name << "' with "
                              << mesh.points.size() << " vertices, material=" << mesh.materialBinding << std::endl;
                }
            }
        }

        // Restore parent transform
        m_currentTranslate = parentT;
        m_currentRotate = parentR;
        m_currentScale = parentS;
    }

    //=========================================================================
    // SCOPE PARSING (container without transforms)
    //=========================================================================

    void parseScope(std::ifstream& file, USDAScene& scene) {
        std::string line;
        int braceDepth = 0;

        // Find opening brace first
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                break;
            }
        }

        if (braceDepth == 0) return;

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // Look for nested definitions
            if (startsWith(trimmed, "def Material")) {
                std::string matName = extractPrimName(trimmed);
                parseMaterial(file, scene, matName);
                // Note: parseMaterial consumes its own braces
            }
            else if (startsWith(trimmed, "def Scope")) {
                parseScope(file, scene);
                // Note: parseScope consumes its own braces
            }
        }
    }

    //=========================================================================
    // MATERIAL PARSING
    //=========================================================================

    void parseMaterial(std::ifstream& file, USDAScene& scene, const std::string& name) {
        USDAMaterial mat;
        mat.name = name;

        std::string line;
        int braceDepth = 0;

        // Find opening brace first
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                break;
            }
        }

        if (braceDepth == 0) return;

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // UsdPreviewSurface inputs
            if (trimmed.find("inputs:diffuseColor") != std::string::npos) {
                Vec3 c = parseVec3Value(trimmed);
                mat.diffuseColor = Color3(c.x, c.y, c.z);
            }
            else if (trimmed.find("inputs:specularColor") != std::string::npos) {
                Vec3 c = parseVec3Value(trimmed);
                mat.specularColor = Color3(c.x, c.y, c.z);
            }
            else if (trimmed.find("inputs:roughness") != std::string::npos) {
                mat.roughness = parseFloatValue(trimmed);
            }
            else if (trimmed.find("inputs:metallic") != std::string::npos) {
                mat.metallic = parseFloatValue(trimmed);
            }
            else if (trimmed.find("inputs:ior") != std::string::npos) {
                mat.ior = parseFloatValue(trimmed);
            }
            else if (trimmed.find("inputs:opacity") != std::string::npos) {
                mat.opacity = parseFloatValue(trimmed);
            }
        }

        scene.materials.push_back(mat);
        std::cout << "[USDAParser] Material '" << name << "': "
                  << "diffuse(" << mat.diffuseColor.r << "," << mat.diffuseColor.g << "," << mat.diffuseColor.b << ") "
                  << "roughness=" << mat.roughness << " metallic=" << mat.metallic << std::endl;
    }

    //=========================================================================
    // SPHERE PRIMITIVE PARSING (actual mathematical sphere)
    //=========================================================================

    void parseCube(std::ifstream& file, USDAScene& scene, const std::string& name) {
        std::cout << "[USDAParser] >>> Entering parseCube for '" << name << "'" << std::endl;

        float size = 2.0f;  // Default size (USD Cube default is 2x2x2)
        std::string materialBinding;

        // Local transforms (can be overridden by inline transforms)
        Vec3 localTranslate(0, 0, 0);
        Vec3 localScale(1, 1, 1);

        std::string line;
        int braceDepth = 0;

        // Find opening brace
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                break;
            }
        }

        if (braceDepth == 0) return;

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // Parse cube properties
            if (trimmed.find("double size") != std::string::npos ||
                trimmed.find("float size") != std::string::npos) {
                size = parseFloatValue(trimmed);
            }
            else if (trimmed.find("material:binding") != std::string::npos) {
                materialBinding = extractMaterialBinding(trimmed);
            }
            // Parse inline transforms (transforms directly on the cube)
            else if (startsWith(trimmed, "float3 xformOp:translate") ||
                     startsWith(trimmed, "double3 xformOp:translate")) {
                localTranslate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:scale") ||
                     startsWith(trimmed, "double3 xformOp:scale")) {
                localScale = parseVec3Value(trimmed);
            }
        }

        // Create mesh from cube primitive
        USDAMesh mesh;
        mesh.name = name;
        mesh.path = m_currentPath;
        // Combine parent transform with local transform
        mesh.translate = Vec3(
            m_currentTranslate.x + localTranslate.x,
            m_currentTranslate.y + localTranslate.y,
            m_currentTranslate.z + localTranslate.z
        );
        mesh.rotate = m_currentRotate;
        mesh.scale = Vec3(
            m_currentScale.x * localScale.x,
            m_currentScale.y * localScale.y,
            m_currentScale.z * localScale.z
        );
        mesh.materialBinding = materialBinding;

        // Half-extent (apply local scale to size)
        float h = size / 2.0f;

        // 8 vertices of a cube
        mesh.points = {
            Vec3(-h, -h, -h), Vec3( h, -h, -h), Vec3( h,  h, -h), Vec3(-h,  h, -h),
            Vec3(-h, -h,  h), Vec3( h, -h,  h), Vec3( h,  h,  h), Vec3(-h,  h,  h)
        };

        // 12 triangles (2 per face)
        mesh.faceVertexCounts = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        mesh.faceVertexIndices = {
            // Front (z+)
            4, 5, 6,  4, 6, 7,
            // Back (z-)
            1, 0, 3,  1, 3, 2,
            // Top (y+)
            3, 7, 6,  3, 6, 2,
            // Bottom (y-)
            0, 1, 5,  0, 5, 4,
            // Right (x+)
            1, 2, 6,  1, 6, 5,
            // Left (x-)
            0, 4, 7,  0, 7, 3
        };

        scene.meshes.push_back(mesh);
        std::cout << "[USDAParser] <<< Cube '" << name << "' size=" << size
                  << " -> mesh with 8 vertices, 12 triangles" << std::endl;
    }

    void parseSphere(std::ifstream& file, USDAScene& scene, const std::string& name) {
        std::cout << "[USDAParser] >>> Entering parseSphere for '" << name << "'" << std::endl;

        USDASphere sphere;
        // If name is generic like "Geom", use parent Xform name instead
        if (name == "Geom" || name == "unnamed" || name.empty()) {
            // Extract parent name from path: /World/Sphere_0/Geom -> Sphere_0
            size_t lastSlash = m_currentPath.rfind('/');
            if (lastSlash != std::string::npos && lastSlash > 0) {
                size_t prevSlash = m_currentPath.rfind('/', lastSlash - 1);
                if (prevSlash != std::string::npos) {
                    sphere.name = m_currentPath.substr(prevSlash + 1, lastSlash - prevSlash - 1);
                } else {
                    sphere.name = name;
                }
            } else {
                sphere.name = name;
            }
            std::cout << "[USDAParser] Using parent name: '" << sphere.name << "'" << std::endl;
        } else {
            sphere.name = name;
        }
        sphere.path = m_currentPath;
        sphere.center = m_currentTranslate;  // Position from parent transform
        sphere.radius = 1.0f;  // Default radius

        // Local transform (can be overridden by inline transforms)
        Vec3 localTranslate(0, 0, 0);
        Vec3 localScale(1, 1, 1);

        std::string line;
        int braceDepth = 0;

        // Find opening brace
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                break;
            }
        }

        if (braceDepth == 0) return;

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            // Parse sphere properties
            if (trimmed.find("double radius") != std::string::npos ||
                trimmed.find("float radius") != std::string::npos) {
                sphere.radius = parseFloatValue(trimmed);
            }
            else if (trimmed.find("material:binding") != std::string::npos) {
                sphere.materialBinding = extractMaterialBinding(trimmed);
            }
            // Parse inline transforms (transforms directly on the sphere)
            else if (startsWith(trimmed, "float3 xformOp:translate") ||
                     startsWith(trimmed, "double3 xformOp:translate")) {
                localTranslate = parseVec3Value(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:scale") ||
                     startsWith(trimmed, "double3 xformOp:scale")) {
                localScale = parseVec3Value(trimmed);
            }
        }

        // Apply local translate to center (added to parent transform)
        sphere.center = Vec3(
            m_currentTranslate.x + localTranslate.x,
            m_currentTranslate.y + localTranslate.y,
            m_currentTranslate.z + localTranslate.z
        );

        // Apply scale to radius (use average for non-uniform scale, then apply parent scale)
        float localScaleAvg = (localScale.x + localScale.y + localScale.z) / 3.0f;
        sphere.radius *= localScaleAvg * m_currentScale.x;

        scene.spheres.push_back(sphere);
        std::cout << "[USDAParser] <<< Sphere '" << name << "' at ("
                  << sphere.center.x << "," << sphere.center.y << "," << sphere.center.z
                  << ") radius=" << sphere.radius << std::endl;
    }

    //=========================================================================
    // LIGHT PARSING
    //=========================================================================

    void parseLight(std::ifstream& file, USDAScene& scene, const std::string& name, USDALightType type) {
        USDALight light;
        light.name = name;
        light.type = type;
        light.position = m_currentTranslate;  // Inherit from parent xform

        std::string line;
        int braceDepth = 0;

        // Find opening brace first
        while (std::getline(file, line)) {
            std::string trimmed = trim(line);
            if (trimmed.empty()) continue;
            if (trimmed[0] == '{' || trimmed.find('{') != std::string::npos) {
                braceDepth = 1;
                break;
            }
        }

        if (braceDepth == 0) return;

        while (std::getline(file, line) && braceDepth > 0) {
            std::string trimmed = trim(line);

            for (char c : trimmed) {
                if (c == '{') braceDepth++;
                else if (c == '}') braceDepth--;
            }

            if (braceDepth <= 0) break;

            if (trimmed.find("inputs:color") != std::string::npos) {
                Vec3 c = parseVec3Value(trimmed);
                light.color = Color3(c.x, c.y, c.z);
            }
            else if (trimmed.find("inputs:intensity") != std::string::npos) {
                light.intensity = parseFloatValue(trimmed);
            }
            else if (trimmed.find("inputs:radius") != std::string::npos) {
                light.radius = parseFloatValue(trimmed);
            }
            else if (startsWith(trimmed, "float3 xformOp:translate") ||
                     startsWith(trimmed, "double3 xformOp:translate")) {
                Vec3 t = parseVec3Value(trimmed);
                light.position = Vec3(
                    m_currentTranslate.x + t.x,
                    m_currentTranslate.y + t.y,
                    m_currentTranslate.z + t.z
                );
            }
            else if (trimmed.find("inputs:angle") != std::string::npos &&
                     trimmed.find("inputs:coneAngle") == std::string::npos &&
                     trimmed.find("inputs:coneFalloff") == std::string::npos) {
                // Direction for distant lights and spotlights
                light.direction = parseVec3Value(trimmed);
            }
            else if (trimmed.find("inputs:coneAngle") != std::string::npos) {
                // Cone angle for spotlights (in degrees)
                light.coneAngle = parseFloatValue(trimmed);
            }
            else if (trimmed.find("inputs:coneFalloff") != std::string::npos) {
                // Falloff softness for spotlights
                light.coneFalloff = parseFloatValue(trimmed);
            }
        }

        scene.lights.push_back(light);
        std::cout << "[USDAParser] Light '" << name << "': "
                  << "pos(" << light.position.x << "," << light.position.y << "," << light.position.z << ") "
                  << "intensity=" << light.intensity << std::endl;
    }

    //=========================================================================
    // VALUE PARSING HELPERS
    //=========================================================================

    std::vector<Vec3> parseVec3Array(const std::string& line) {
        std::vector<Vec3> result;

        // Find array content between [ and ] AFTER the '=' sign
        // This avoids matching the [] in type declarations like 'float3[]'
        size_t eqPos = line.find('=');
        size_t searchStart = (eqPos != std::string::npos) ? eqPos : 0;
        size_t start = line.find('[', searchStart);
        size_t end = line.rfind(']');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            return result;
        }

        std::string content = line.substr(start + 1, end - start - 1);

        // Parse (x, y, z) tuples
        size_t pos = 0;
        while ((pos = content.find('(', pos)) != std::string::npos) {
            size_t endParen = content.find(')', pos);
            if (endParen == std::string::npos) break;

            std::string tuple = content.substr(pos + 1, endParen - pos - 1);
            Vec3 v = parseTuple3(tuple);
            result.push_back(v);

            pos = endParen + 1;
        }

        return result;
    }

    std::vector<Vec2> parseVec2Array(const std::string& line) {
        std::vector<Vec2> result;

        // Find array content between [ and ] AFTER the '=' sign
        size_t eqPos = line.find('=');
        size_t searchStart = (eqPos != std::string::npos) ? eqPos : 0;
        size_t start = line.find('[', searchStart);
        size_t end = line.rfind(']');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            return result;
        }

        std::string content = line.substr(start + 1, end - start - 1);

        size_t pos = 0;
        while ((pos = content.find('(', pos)) != std::string::npos) {
            size_t endParen = content.find(')', pos);
            if (endParen == std::string::npos) break;

            std::string tuple = content.substr(pos + 1, endParen - pos - 1);
            Vec2 v = parseTuple2(tuple);
            result.push_back(v);

            pos = endParen + 1;
        }

        return result;
    }

    std::vector<int> parseIntArray(const std::string& line) {
        std::vector<int> result;

        // Find array content between [ and ] AFTER the '=' sign
        // This avoids matching the [] in type declarations like 'int[]'
        size_t eqPos = line.find('=');
        size_t searchStart = (eqPos != std::string::npos) ? eqPos : 0;
        size_t start = line.find('[', searchStart);
        size_t end = line.rfind(']');
        if (start == std::string::npos || end == std::string::npos || end <= start) {
            return result;
        }

        std::string content = line.substr(start + 1, end - start - 1);

        std::stringstream ss(content);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token = trim(token);
            if (!token.empty()) {
                try {
                    result.push_back(std::stoi(token));
                } catch (...) {}
            }
        }

        return result;
    }

    Vec3 parseTuple3(const std::string& tuple) {
        Vec3 v;
        std::stringstream ss(tuple);
        std::string token;
        int i = 0;
        while (std::getline(ss, token, ',') && i < 3) {
            token = trim(token);
            try {
                float val = std::stof(token);
                if (i == 0) v.x = val;
                else if (i == 1) v.y = val;
                else if (i == 2) v.z = val;
            } catch (...) {}
            i++;
        }
        return v;
    }

    Vec2 parseTuple2(const std::string& tuple) {
        Vec2 v;
        std::stringstream ss(tuple);
        std::string token;
        int i = 0;
        while (std::getline(ss, token, ',') && i < 2) {
            token = trim(token);
            try {
                float val = std::stof(token);
                if (i == 0) v.u = val;
                else if (i == 1) v.v = val;
            } catch (...) {}
            i++;
        }
        return v;
    }

    Vec3 parseVec3Value(const std::string& line) {
        // Handle: float3 xformOp:translate = (1, 2, 3)
        // Or: color3f inputs:diffuseColor = (0.8, 0.2, 0.1)
        size_t start = line.find('(');
        size_t end = line.find(')');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            return parseTuple3(line.substr(start + 1, end - start - 1));
        }
        return Vec3();
    }

    float parseFloatValue(const std::string& line) {
        // Handle: float inputs:roughness = 0.5
        size_t eq = line.find('=');
        if (eq != std::string::npos) {
            std::string val = trim(line.substr(eq + 1));
            // Remove trailing characters
            size_t end = val.find_first_of(" \t\n\r");
            if (end != std::string::npos) {
                val = val.substr(0, end);
            }
            try {
                return std::stof(val);
            } catch (...) {}
        }
        return 0.0f;
    }

    //=========================================================================
    // STRING HELPERS
    //=========================================================================

    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        return s.substr(start, end - start + 1);
    }

    // Strip comments from a line (everything after # outside of strings)
    std::string stripComment(const std::string& s) {
        bool inString = false;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '"') inString = !inString;
            else if (s[i] == '#' && !inString) {
                return s.substr(0, i);
            }
        }
        return s;
    }

    bool startsWith(const std::string& s, const std::string& prefix) {
        return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
    }

    std::string extractQuotedString(const std::string& line) {
        size_t start = line.find('"');
        if (start == std::string::npos) return "";
        size_t end = line.find('"', start + 1);
        if (end == std::string::npos) return "";
        return line.substr(start + 1, end - start - 1);
    }

    std::string extractPrimName(const std::string& line) {
        // def Mesh "MyMesh" { -> MyMesh
        size_t start = line.find('"');
        if (start == std::string::npos) return "unnamed";
        size_t end = line.find('"', start + 1);
        if (end == std::string::npos) return "unnamed";
        return line.substr(start + 1, end - start - 1);
    }

    std::string extractAttributeKey(const std::string& line) {
        // "point3f[] points = [" -> "points"
        size_t space = line.find(' ');
        if (space == std::string::npos) return line;
        size_t eq = line.find('=');
        if (eq == std::string::npos) eq = line.length();
        return trim(line.substr(space + 1, eq - space - 1));
    }

    //=========================================================================
    // OBJ PAYLOAD LOADING
    //=========================================================================

    // Extract payload path from "payload = @filename.obj@" syntax
    std::string extractPayloadPath(const std::string& line) {
        size_t atStart = line.find('@');
        if (atStart == std::string::npos) return "";
        size_t atEnd = line.find('@', atStart + 1);
        if (atEnd == std::string::npos) return "";
        return line.substr(atStart + 1, atEnd - atStart - 1);
    }

    // Load OBJ file and populate mesh data
    bool loadOBJ(const std::string& filename, USDAMesh& mesh) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[USDAParser] Could not open OBJ file: " << filename << std::endl;
            return false;
        }

        std::cout << "[USDAParser] Loading OBJ: " << filename << std::endl;

        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<std::vector<int>> faceVertexIndices;  // Each face has indices
        std::vector<std::vector<int>> faceNormalIndices;  // Each face has normal indices

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                // Vertex position
                float x, y, z;
                iss >> x >> y >> z;
                positions.push_back(Vec3(x, y, z));
            }
            else if (prefix == "vn") {
                // Vertex normal
                float x, y, z;
                iss >> x >> y >> z;
                normals.push_back(Vec3(x, y, z));
            }
            else if (prefix == "f") {
                // Face - can be "v", "v/vt", "v/vt/vn", or "v//vn"
                std::vector<int> vertIndices;
                std::vector<int> normIndices;
                std::string vertexData;
                while (iss >> vertexData) {
                    int vi = 0, vni = 0;

                    // Parse vertex/texcoord/normal indices
                    size_t slash1 = vertexData.find('/');
                    if (slash1 == std::string::npos) {
                        // Just vertex index
                        vi = std::stoi(vertexData);
                    } else {
                        vi = std::stoi(vertexData.substr(0, slash1));
                        size_t slash2 = vertexData.find('/', slash1 + 1);
                        if (slash2 == std::string::npos) {
                            // v/vt format - texture coords ignored (not used)
                        } else {
                            // v/vt/vn or v//vn format
                            // (texture coords between slashes ignored - not used)
                            if (slash2 + 1 < vertexData.length()) {
                                vni = std::stoi(vertexData.substr(slash2 + 1));
                            }
                        }
                    }

                    // OBJ indices are 1-based, convert to 0-based
                    vertIndices.push_back(vi - 1);
                    if (vni > 0) normIndices.push_back(vni - 1);
                }

                if (vertIndices.size() >= 3) {
                    faceVertexIndices.push_back(vertIndices);
                    if (!normIndices.empty()) {
                        faceNormalIndices.push_back(normIndices);
                    }
                }
            }
        }

        file.close();

        // Convert to USDAMesh format
        mesh.points = positions;

        // Build face data - triangulate if needed
        for (size_t fi = 0; fi < faceVertexIndices.size(); fi++) {
            const auto& face = faceVertexIndices[fi];
            mesh.faceVertexCounts.push_back(static_cast<int>(face.size()));
            for (int idx : face) {
                mesh.faceVertexIndices.push_back(idx);
            }
        }

        // Handle normals - if per-face-vertex normals, we may need to duplicate vertices
        // For simplicity, just use face normals if no vertex normals provided
        if (!normals.empty() && !faceNormalIndices.empty()) {
            // OBJ has separate normal indices - for now just store them
            // The renderer will compute normals if needed
            mesh.normals.clear();  // Let the renderer compute smooth normals
        }

        std::cout << "[USDAParser] OBJ loaded: " << mesh.points.size() << " vertices, "
                  << mesh.faceVertexCounts.size() << " faces" << std::endl;

        return !mesh.points.empty();
    }

    std::string extractMaterialPath(const std::string& line) {
        // rel material:binding = </Materials/Metal>
        size_t start = line.find('<');
        size_t end = line.find('>');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string path = line.substr(start + 1, end - start - 1);
            // Return just the material name
            size_t lastSlash = path.rfind('/');
            if (lastSlash != std::string::npos) {
                return path.substr(lastSlash + 1);
            }
            return path;
        }
        return "";
    }

    std::string extractMaterialBinding(const std::string& line) {
        // Same as extractMaterialPath - alias for clarity
        return extractMaterialPath(line);
    }
};

} // namespace usd

#endif // USDA_PARSER_H
