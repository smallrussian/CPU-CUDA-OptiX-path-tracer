
#ifndef VIEWPORT_H
#define VIEWPORT_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_1_Core>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QImage>
#include <QThread>

#include <memory>
#include <atomic>
#include "cuda/vec3.h"
#include "graphics/MouseCamera.h"
#include "graphics/Mesh.h"
#include "graphics/Light.h"
#include "graphics/Material.h"
#include "raytracer/renderer.h"
#include "raytracer/scene_builder.h"
#include "usd/USDAParser.h"

class QTimer;
class CudaRenderer;
class CudaGLInterlop;
class OptixRenderer;

namespace graphics {

// Render mode enum
enum class RenderMode {
    OPENGL,          // OpenGL rasterization (Phong shading)
    CPU_RAYTRACER,   // CPU path tracing (RTiOW-based, batch only)
    CUDA_SOFTWARE,   // Software ray tracing on CUDA cores (real-time)
    OPTIX_HARDWARE   // Hardware ray tracing on RT cores (real-time)
};

class Viewport : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit Viewport(QWidget *parent = nullptr);
    virtual ~Viewport();

public slots:
    void onTimeout();

    // Render settings
    void setSamplesPerPixel(int samples);
    void setMaxDepth(int depth);
    int getSamplesPerPixel() const { return samplesPerPixel; }
    int getMaxDepth() const { return maxDepth; }

    // CUDA max depth (from config file, runtime configurable)
    void setCudaMaxDepth(int depth);
    int getCudaMaxDepth() const;

    // Scene I/O
    bool exportSceneToUSDA(const QString& filename);
    bool loadSceneFromUSDA(const QString& filename);

    // Render mode
    void toggleRenderMode();
    RenderMode getRenderMode() const { return renderMode; }
    QString getRenderModeString() const;

    // Denoiser (OptiX only)
    void toggleDenoiser();
    bool isDenoiserEnabled() const;

    // Scene loading (for OpenGL/CPU modes)
    bool loadOBJ(const std::string& filename);
    bool loadUSD(const std::string& filename);
    bool loadScene(const std::string& filename);  // Dispatches to loadOBJ or loadUSD
    void setPendingSceneFile(const std::string& filename);
    std::string getCurrentSceneFile() const { return currentSceneFile; }

    // Batch rendering (CPU mode only)
    void startBatchRender(const rt::RenderSettings& settings);
    void cancelBatchRender();
    bool isBatchRenderingActive() const { return batchRendering; }

    // Light/Material modification
    void updateLight(int index, float r, float g, float b, float intensity);
    void updateMaterial(int objectIndex, int materialType, float r, float g, float b, float param);
    int getLightCount() const;
    QStringList getObjectNames() const;
    void getLightValues(int index, float& r, float& g, float& b, float& intensity) const;
    void getMaterialValues(int index, int& type, float& r, float& g, float& b, float& param) const;

signals:
    void batchRenderProgress(int currentFrame, int totalFrames);
    void batchRenderFrameComplete(double frameTimeSeconds);
    void batchRenderComplete(bool success, const QString& message);

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

private:
    void initQuad();
    void initShader();
    void updateCameraFromOrbit();
    vec3 getCameraPosition() const;
    void rebuildCPURayTracingScene(bool useBVH = true);

    QTimer* timer;

    // OpenGL resources
    GLuint quadVAO;
    GLuint quadVBO;
    GLuint displayShader;

    // CUDA Rendering
    CudaRenderer* cudaRenderer;
    CudaGLInterlop* cudaInterop;

    // OptiX Rendering
    OptixRenderer* optixRenderer;
    RenderMode renderMode;

    // Render settings
    int samplesPerPixel;
    int maxDepth;

    // Camera orbit state
    vec3 camLookAt;       // Target point (center of scene)
    float camDistance;    // Distance from target
    float camYaw;         // Horizontal angle (radians)
    float camPitch;       // Vertical angle (radians)
    float camVFOV;        // Field of view
    float camAperture;    // Depth of field aperture
    float camFocusDist;   // Focus distance

    // Mouse tracking
    QPoint lastMousePos;
    bool isDragging;

    bool initialized;

    // ============= CPU Ray Tracer Members (from Part 1) =============
    // OpenGL rendering (for OPENGL mode)
    std::unique_ptr<MouseCameraf> glCamera;
    std::vector<std::unique_ptr<Mesh>> meshes;
    std::vector<Lightf> lights;
    std::vector<std::shared_ptr<Material<float>>> sceneMaterials;

    // CPU Ray tracing members (for CPU_RAYTRACER mode)
    std::unique_ptr<rt::Renderer> cpuRayTracer;
    std::shared_ptr<rt::hittable> cpuScene;  // RTiOW hittable (BVH or hittable_list)
    rt::scene_builder cpuSceneBuilder;
    QImage rayTracedImage;

    // Sphere primitives storage (for ray tracing - RTiOW style)
    struct SphereData {
        rt::point3 center;
        double radius;
        rt::MaterialInfo material;
        std::string name;  // Object name from scene
    };
    std::vector<SphereData> spherePrimitives;
    size_t numUSDMeshes = 0;  // Count of actual USD mesh geometry (excludes GL sphere approximations)

    // Scene file tracking
    std::string currentSceneFile;
    std::string pendingSceneFile;  // File to load on initializeGL
    bool sceneLoaded = false;

    // Batch rendering state (CPU mode only)
    QThread* renderThread = nullptr;
    std::atomic<bool> batchRendering{false};
};

}

#endif
