
#ifndef VIEWPORTGL_H
#define VIEWPORTGL_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_1_Core>
#include <QMatrix4x4>
#include <QImage>
#include <QThread>

#include <memory>
#include <atomic>
#include "graphics/MouseCamera.h"
#include "graphics/Mesh.h"
#include "graphics/Light.h"
#include "graphics/Material.h"
#include "raytracer/renderer.h"
#include "raytracer/scene_builder.h"
#include "usd/USDAParser.h"

class QTimer;

namespace graphics {

// Forward declare rt types used
using rt::RenderSettings;
using rt::MaterialInfo;
using rt::point3;
using rt::color;
using rt::hittable;
using rt::scene_builder;
using rt::Renderer;

class ViewportGL : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit ViewportGL(QWidget *parent = nullptr);
    virtual ~ViewportGL();

public slots:
    void onTimeout();

    // Scene loading
    bool loadOBJ(const std::string& filename);
    bool loadUSD(const std::string& filename);
    bool loadScene(const std::string& filename);  // Dispatches to loadOBJ or loadUSD
    void setPendingSceneFile(const std::string& filename);  // Set file to load on init

    // Render settings
    void setSamplesPerPixel(int samples);
    void setMaxDepth(int depth);
    int getSamplesPerPixel() const { return samplesPerPixel; }
    int getMaxDepth() const { return maxDepth; }

    // Batch rendering
    void startBatchRender(const RenderSettings& settings);
    void cancelBatchRender();
    bool isBatchRenderingActive() const { return batchRendering; }

    // Get current scene file path
    std::string getCurrentSceneFile() const { return currentSceneFile; }

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

    void mousePressEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;

private:
    void updateCameraFromOrbit();

    // Orbit camera state (matches CUDA version)
    bool isDragging = false;
    QPoint lastMousePos;
    float camYaw = 0.0f;
    float camPitch = 0.3f;
    float camDistance = 13.0f;
    float camTargetY = 1.0f;
    void rebuildRayTracingScene(bool useBVH = true);

    QTimer* timer;
    std::unique_ptr<MouseCameraf> camera;
    std::vector<std::unique_ptr<Mesh>> meshes;

    // Ray tracing members
    QImage rayTracedImage;
    std::unique_ptr<Renderer> rayTracer;
    std::shared_ptr<hittable> scene;
    std::vector<Lightf> lights;

    // Scene builder for constructing RT scene from meshes
    scene_builder sceneBuilder;

    // GL material storage (for OpenGL Phong rendering)
    std::vector<std::shared_ptr<Material<float>>> sceneMaterials;
    std::vector<size_t> meshMaterialIndices;  // Maps mesh index to material index

    // RT material storage (for CPU ray tracing - stores metallic/roughness/glass info)
    std::vector<MaterialInfo> meshRTMaterials;  // Maps mesh index to RT material
    std::vector<std::string> meshNames;  // Mesh names for material controls

    // Sphere primitives storage (for ray tracing - RTiOW style)
    struct SphereData {
        point3 center;
        double radius;
        MaterialInfo material;
        std::string name;  // Object name from scene
    };
    std::vector<SphereData> spherePrimitives;
    size_t numUSDMeshes = 0;

    // Render settings
    int samplesPerPixel;
    int maxDepth;

    // Current scene file
    std::string currentSceneFile;
    std::string pendingSceneFile;
    bool sceneLoaded;

    // Batch rendering state
    QThread* renderThread = nullptr;
    std::atomic<bool> batchRendering{false};
};

}

#endif
