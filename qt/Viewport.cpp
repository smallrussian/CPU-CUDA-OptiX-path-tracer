
#include "Viewport.h"
#include <QTimer>
#include <QDir>
#include <QElapsedTimer>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

#include <QOpenGLVersionFunctionsFactory>
#include <QSurfaceFormat>

#include "cuda/CudaRenderer.h"
#include "cuda/CudaGLInterlop.h"
#include "cuda/OptixRenderer.h"


namespace graphics {

// fullscreen quad verticies 
static float quadVertices[] = { 
	// positions   // texCoords
	-1.0f,  1.0f,  0.0f, 1.0f,
	-1.0f, -1.0f,  0.0f, 0.0f,
	 1.0f, -1.0f,  1.0f, 0.0f,

	-1.0f,  1.0f,  0.0f, 1.0f,
	 1.0f, -1.0f,  1.0f, 0.0f,
	 1.0f,  1.0f,  1.0f, 1.0f
};

// Simple vertex shader source
static const char* vertexShaderSource = R"(
#version 410 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";


// Simple fragment shader source
static const char* fragmentShaderSource = R"(
#version 410 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;
void main() {
    FragColor = texture(screenTexture, TexCoord);
}
)";

Viewport::Viewport(QWidget *parent) : QOpenGLWidget(parent),
	timer(nullptr),
    quadVAO(0),
    quadVBO(0),
    displayShader(0),
    cudaRenderer(nullptr),
    cudaInterop(nullptr),
    optixRenderer(nullptr),
    renderMode(RenderMode::CUDA_SOFTWARE),
	samplesPerPixel(10),
	maxDepth(5),
	camLookAt(0, 0, 0),
	camDistance(13.5f),
	camYaw(1.34f),        // atan2(13, 3) ~= 1.34 rad
	camPitch(0.15f),      // asin(2/13.5) ~= 0.15 rad
	camVFOV(30.0f),
	camAperture(0.0f),  // DOF disabled
	camFocusDist(10.0f),
	lastMousePos(0, 0),
	isDragging(false),
	initialized(false)
{
	// Disable vsync - set swap interval to 0
	QSurfaceFormat format = QSurfaceFormat::defaultFormat();
	format.setSwapInterval(0);  // 0 = no vsync, 1 = vsync on
	setFormat(format);

	timer = new QTimer(this);
	timer->setInterval(0); // No delay - render as fast as possible
	connect(timer, &QTimer::timeout, this, &Viewport::onTimeout);
}

Viewport::~Viewport() {
		makeCurrent();

		if (cudaRenderer) {
			cudaRenderer->cleanup();
			delete cudaRenderer;
			cudaRenderer = nullptr;
		}

		if (optixRenderer) {
			optixRenderer->cleanup();
			delete optixRenderer;
			optixRenderer = nullptr;
		}

		if (cudaInterop) {
			cudaInterop->cleanup();
			delete cudaInterop;
			cudaInterop = nullptr;
		}

		// Use Qt OpenGL functions for cleanup
		auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
			QOpenGLContext::currentContext());
		if (f) {
			if (quadVAO) f->glDeleteVertexArrays(1, &quadVAO);
			if (quadVBO) f->glDeleteBuffers(1, &quadVBO);
			if (displayShader) f->glDeleteProgram(displayShader);
		}

		doneCurrent();
		
}

void Viewport::onTimeout() {
    // Don't update viewport during GPU batch rendering (CUDA/OptiX)
    // CPU batch rendering runs on separate thread, so viewport can continue
    if (batchRendering) {
        return;
    }
    this->update();
}

void Viewport::initializeGL() {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
        QOpenGLContext::currentContext());
    f->initializeOpenGLFunctions();
    f->glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    // Initialize fullscreen quad
    initQuad();
    // Initialize texture display shader
    initShader();
    // Initialize CUDA renderer
    cudaRenderer = new CudaRenderer();
    if (!cudaRenderer->initialize(width(), height())) {
        std::cerr << "[Viewport] Failed to initialize CUDA renderer" << std::endl;
        return;
    }
    // Initialize CUDA-GL interop
    cudaInterop = new CudaGLInterlop();
    if (!cudaInterop->initialize(width(), height())) {
        std::cerr << "[Viewport] Failed to initialize CUDA-GL interop" << std::endl;
        return;
    }

    // Initialize OptiX renderer
    optixRenderer = new OptixRenderer();
    if (!optixRenderer->initialize(width(), height())) {
        std::cerr << "[Viewport] Failed to initialize OptiX renderer" << std::endl;
        // Not fatal - can still use CUDA renderer
    } else {
        // Build OptiX GAS using same sphere and triangle data as CUDA renderer (with materials)
        const auto& spheres = cudaRenderer->getSpheres();
        const auto& triangles = cudaRenderer->getTriangles();
        const auto& meshes = cudaRenderer->getMeshes();
        optixRenderer->buildAccel(spheres, triangles, meshes);

        // Sync camera with CUDA renderer's initial view
        vec3 lookfrom = getCameraPosition();
        optixRenderer->updateCamera(lookfrom, camLookAt, vec3(0, 1, 0), camVFOV);
        std::cout << "[Viewport] OptiX renderer initialized with " << spheres.size() << " spheres and " << triangles.size() << " triangles" << std::endl;
    }

    // Initialize CPU ray tracer (for CPU_RAYTRACER mode batch rendering)
    cpuRayTracer = std::make_unique<rt::Renderer>();
    std::cout << "[Viewport] CPU ray tracer initialized" << std::endl;

    initialized = true;
    std::cout << "[Viewport] Initialized " << width() << "x" << height() << " (Mode: CUDA)" << std::endl;

    // Load pending scene file if passed from OpenGL app (overrides default scene)
    std::cout << "[Viewport] pendingSceneFile = '" << pendingSceneFile << "'" << std::endl;
    if (!pendingSceneFile.empty()) {
        std::cout << "[Viewport] Loading scene from pending file: " << pendingSceneFile << std::endl;
        loadScene(pendingSceneFile);
        pendingSceneFile.clear();
        std::cout << "[Viewport] Scene loaded, CUDA spheres = " << (cudaRenderer ? cudaRenderer->getSpheres().size() : 0) << std::endl;
    } else {
        // No pending file - just build from current CUDA state (may be empty)
        std::cout << "[Viewport] No pending scene file, building from current CUDA state" << std::endl;
        rebuildCPURayTracingScene(true);
    }

    timer->start();
}

void Viewport::initQuad() {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
        QOpenGLContext::currentContext());
    f->glGenVertexArrays(1, &quadVAO);
    f->glGenBuffers(1, &quadVBO);
    f->glBindVertexArray(quadVAO);
    f->glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    f->glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    // Position attribute
    f->glEnableVertexAttribArray(0);
    f->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    // TexCoord attribute
    f->glEnableVertexAttribArray(1);
    f->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    f->glBindVertexArray(0);
}


void Viewport::initShader() {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
        QOpenGLContext::currentContext());
    // Compile vertex shader
    GLuint vertexShader = f->glCreateShader(GL_VERTEX_SHADER);
    f->glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    f->glCompileShader(vertexShader);
    // Check vertex shader
    GLint success;
    f->glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        f->glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "[Viewport] Vertex shader error: " << infoLog << std::endl;
    }
    // Compile fragment shader
    GLuint fragmentShader = f->glCreateShader(GL_FRAGMENT_SHADER);
    f->glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    f->glCompileShader(fragmentShader);
    // Check fragment shader
    f->glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        f->glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "[Viewport] Fragment shader error: " << infoLog << std::endl;
    }
    // Link program
    displayShader = f->glCreateProgram();
    f->glAttachShader(displayShader, vertexShader);
    f->glAttachShader(displayShader, fragmentShader);
    f->glLinkProgram(displayShader);
    // Check program
    f->glGetProgramiv(displayShader, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        f->glGetProgramInfoLog(displayShader, 512, nullptr, infoLog);
        std::cerr << "[Viewport] Shader program error: " << infoLog << std::endl;
    }
    f->glDeleteShader(vertexShader);
    f->glDeleteShader(fragmentShader);
}



void Viewport::resizeGL(int w, int h) {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
        QOpenGLContext::currentContext());
    f->glViewport(0, 0, w, h);
    if (initialized && cudaRenderer && cudaInterop) {
        cudaRenderer->resize(w, h);
        cudaInterop->resize(w, h);
    }
    if (initialized && optixRenderer) {
        optixRenderer->resize(w, h);
    }
}

void Viewport::paintGL() {
    if (!initialized) return;
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(
        QOpenGLContext::currentContext());

    int fbWidth = 0, fbHeight = 0;

    if (renderMode == RenderMode::CUDA_SOFTWARE) {
        // 1. Render with CUDA
        cudaRenderer->render(samplesPerPixel);
        vec3* framebuffer = cudaRenderer->getFramebuffer();
        fbWidth = cudaRenderer->getWidth();
        fbHeight = cudaRenderer->getHeight();
        if (!framebuffer) return;
        // 2. Copy vec3 framebuffer to GL texture
        cudaInterop->copyFramebufferToTexture(framebuffer, fbWidth, fbHeight);
    } else if (renderMode == RenderMode::OPTIX_HARDWARE && optixRenderer) {
        // 1. Render with OptiX
        optixRenderer->render(samplesPerPixel);
        float4* framebuffer = optixRenderer->getFramebuffer();
        fbWidth = optixRenderer->getWidth();
        fbHeight = optixRenderer->getHeight();
        if (!framebuffer) return;
        // 2. Copy float4 framebuffer to GL texture
        cudaInterop->copyFramebufferToTexture(framebuffer, fbWidth, fbHeight);
    } else {
        return;
    }

    // 3. Draw fullscreen quad with texture
    f->glClear(GL_COLOR_BUFFER_BIT);
    f->glDisable(GL_DEPTH_TEST);
    f->glUseProgram(displayShader);
    f->glActiveTexture(GL_TEXTURE0);
    f->glBindTexture(GL_TEXTURE_2D, cudaInterop->getTexture());
    f->glBindVertexArray(quadVAO);
    f->glDrawArrays(GL_TRIANGLES, 0, 6);
    f->glBindVertexArray(0);
}
void Viewport::setSamplesPerPixel(int samples) {
    samplesPerPixel = (std::max)(1, samples);
    std::cout << "[Viewport] Samples per pixel: " << samplesPerPixel << std::endl;
    this->update();
}

void Viewport::setMaxDepth(int depth) {
    maxDepth = (std::max)(1, depth);
    std::cout << "[Viewport] Max depth: " << maxDepth << std::endl;
    this->update();
}

void Viewport::setCudaMaxDepth(int depth) {
    if (cudaRenderer) {
        cudaRenderer->setMaxDepth(depth);
        std::cout << "[Viewport] CUDA max depth set to: " << depth << std::endl;
    }
}

int Viewport::getCudaMaxDepth() const {
    if (cudaRenderer) {
        return cudaRenderer->getMaxDepth();
    }
    return 10;  // Default
}

bool Viewport::exportSceneToUSDA(const QString& filename) {
    if (!cudaRenderer) return false;
    return cudaRenderer->exportSceneToUSDA(filename.toStdString());
}

bool Viewport::loadSceneFromUSDA(const QString& filename) {
    if (!cudaRenderer) return false;
    bool result = cudaRenderer->loadSceneFromUSDA(filename.toStdString());
    if (result) {
        // Reset camera to default view
        camLookAt = vec3(0, 0, 0);
        camDistance = 13.5f;
        camYaw = 1.34f;
        camPitch = 0.15f;
        updateCameraFromOrbit();

        // Rebuild OptiX scene with new sphere, triangle, and light data
        if (optixRenderer) {
            const auto& spheres = cudaRenderer->getSpheres();
            const auto& triangles = cudaRenderer->getTriangles();
            const auto& meshes = cudaRenderer->getMeshes();
            const auto& lights = cudaRenderer->getLights();
            std::cout << "[Viewport] Rebuilding OptiX with " << spheres.size() << " spheres, " << triangles.size() << " triangles" << std::endl;
            bool accelBuilt = optixRenderer->buildAccel(spheres, triangles, meshes);
            if (!accelBuilt) {
                std::cerr << "[Viewport] ERROR: OptiX buildAccel failed!" << std::endl;
            } else {
                optixRenderer->setLights(lights);
                std::cout << "[Viewport] OptiX scene rebuilt with " << spheres.size() << " spheres, " << triangles.size() << " triangles, and " << lights.size() << " lights" << std::endl;
            }
        }

        // Rebuild CPU scene to populate spherePrimitives for material controls
        rebuildCPURayTracingScene(true);
    }
    return result;
}

vec3 Viewport::getCameraPosition() const {
    // Spherical to Cartesian conversion
    float x = camDistance * cosf(camPitch) * sinf(camYaw);
    float y = camDistance * sinf(camPitch);
    float z = camDistance * cosf(camPitch) * cosf(camYaw);
    return camLookAt + vec3(x, y, z);
}

void Viewport::updateCameraFromOrbit() {
    if (!initialized || !cudaRenderer) return;

    vec3 lookfrom = getCameraPosition();

    // Use camera distance as focus distance so the look-at point is always in focus
    float dynamicFocusDist = camDistance;

    cudaRenderer->updateCamera(
        lookfrom, camLookAt, vec3(0, 1, 0),
        camVFOV, camAperture, dynamicFocusDist
    );

    // Also update OptiX camera for seamless mode switching (with DOF)
    if (optixRenderer) {
        optixRenderer->updateCamera(lookfrom, camLookAt, vec3(0, 1, 0), camVFOV, camAperture, dynamicFocusDist);
    }
}

void Viewport::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::RightButton) {
        isDragging = true;
        lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
}

void Viewport::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::RightButton) {
        isDragging = false;
        setCursor(Qt::ArrowCursor);
    }
}

void Viewport::mouseMoveEvent(QMouseEvent *event) {
    if (!isDragging) return;

    QPoint delta = event->pos() - lastMousePos;
    lastMousePos = event->pos();

    // Rotate camera based on mouse movement
    float sensitivity = 0.005f;
    camYaw -= delta.x() * sensitivity;
    camPitch += delta.y() * sensitivity;

    // Clamp pitch to avoid flipping
    float maxPitch = 1.5f;  // ~86 degrees
    if (camPitch > maxPitch) camPitch = maxPitch;
    if (camPitch < -maxPitch) camPitch = -maxPitch;

    updateCameraFromOrbit();
}

void Viewport::wheelEvent(QWheelEvent *event) {
    // Zoom in/out
    float zoomFactor = 0.001f;
    camDistance -= event->angleDelta().y() * zoomFactor * camDistance;

    // Clamp distance
    if (camDistance < 1.0f) camDistance = 1.0f;
    if (camDistance > 100.0f) camDistance = 100.0f;

    updateCameraFromOrbit();
}

void Viewport::toggleRenderMode() {
    if (renderMode == RenderMode::CUDA_SOFTWARE) {
        if (optixRenderer && cudaRenderer) {
            renderMode = RenderMode::OPTIX_HARDWARE;

            // Rebuild OptiX scene from CUDA data in case it wasn't built or is stale
            const auto& spheres = cudaRenderer->getSpheres();
            const auto& triangles = cudaRenderer->getTriangles();
            const auto& meshes = cudaRenderer->getMeshes();
            const auto& lights = cudaRenderer->getLights();

            if (!spheres.empty() || !triangles.empty()) {
                std::cout << "[Viewport] Rebuilding OptiX scene: " << spheres.size() << " spheres, " << triangles.size() << " triangles" << std::endl;
                bool accelBuilt = optixRenderer->buildAccel(spheres, triangles, meshes);
                if (accelBuilt) {
                    optixRenderer->setLights(lights);
                } else {
                    std::cerr << "[Viewport] ERROR: OptiX buildAccel failed!" << std::endl;
                }
            }

            // Sync camera to OptiX with all parameters (FOV, aperture, focus)
            vec3 lookfrom = getCameraPosition();
            optixRenderer->updateCamera(lookfrom, camLookAt, vec3(0, 1, 0), camVFOV, camAperture, camFocusDist);
            std::cout << "[Viewport] Switched to OptiX (RT Cores)" << std::endl;
        } else {
            std::cerr << "[Viewport] OptiX not available" << std::endl;
        }
    } else {
        renderMode = RenderMode::CUDA_SOFTWARE;
        std::cout << "[Viewport] Switched to CUDA (Software)" << std::endl;
    }
    this->update();
}

QString Viewport::getRenderModeString() const {
    switch (renderMode) {
        case RenderMode::CUDA_SOFTWARE:
            return "CUDA (Software RT)";
        case RenderMode::OPTIX_HARDWARE:
            return "OptiX (RT Cores)";
        default:
            return "Unknown";
    }
}

void Viewport::toggleDenoiser() {
    if (optixRenderer && renderMode == RenderMode::OPTIX_HARDWARE) {
        optixRenderer->toggleDenoiser();
        std::cout << "[Viewport] Denoiser: " << (optixRenderer->isDenoiserEnabled() ? "ON" : "OFF") << std::endl;
        this->update();
    }
}

bool Viewport::isDenoiserEnabled() const {
    if (optixRenderer) {
        return optixRenderer->isDenoiserEnabled();
    }
    return false;
}

// ============= CPU Ray Tracer Methods (Phase 3 stubs) =============

bool Viewport::loadOBJ(const std::string& filename) {
    // TODO: Implement OBJ loading for GL/CPU modes
    std::clog << "[Viewport] loadOBJ not yet implemented: " << filename << std::endl;
    return false;
}

bool Viewport::loadUSD(const std::string& filename) {
    return loadSceneFromUSDA(QString::fromStdString(filename));
}

bool Viewport::loadScene(const std::string& filename) {
    // Dispatch to appropriate loader based on extension
    if (filename.find(".obj") != std::string::npos || filename.find(".OBJ") != std::string::npos) {
        return loadOBJ(filename);
    } else if (filename.find(".usda") != std::string::npos || filename.find(".USDA") != std::string::npos) {
        return loadUSD(filename);
    }
    std::cerr << "[Viewport] Unknown file format: " << filename << std::endl;
    return false;
}

void Viewport::setPendingSceneFile(const std::string& filename) {
    pendingSceneFile = filename;
    std::clog << "[Viewport] Pending scene file set: " << filename << std::endl;
}

void Viewport::startBatchRender(const rt::RenderSettings& settings) {
    if (batchRendering) {
        std::cerr << "[Viewport] Batch render already in progress" << std::endl;
        return;
    }

    std::clog << "[Viewport] Starting batch render" << std::endl;
    std::clog << "  Mode: " << (settings.render_mode == rt::BatchRenderMode::CUDA_SOFTWARE ? "CUDA" :
                               settings.render_mode == rt::BatchRenderMode::OPTIX_HARDWARE ? "OptiX" : "CPU") << std::endl;
    std::clog << "  Frames: " << settings.start_frame << "-" << settings.end_frame << std::endl;
    std::clog << "  Resolution: " << settings.width << "x" << settings.height << std::endl;
    std::clog << "  Samples: " << settings.samples_per_pixel << std::endl;

    // Create output directory
    QDir dir;
    QString outputPath = settings.output_directory;
    if (!dir.mkpath(outputPath)) {
        std::cerr << "[Viewport] Failed to create output directory: " << outputPath.toStdString() << std::endl;
        emit batchRenderComplete(false, "Failed to create output directory");
        return;
    }

    batchRendering = true;
    int totalFrames = settings.end_frame - settings.start_frame + 1;

    // GPU batch rendering (CUDA/OptiX)
    if (settings.render_mode == rt::BatchRenderMode::CUDA_SOFTWARE || settings.render_mode == rt::BatchRenderMode::OPTIX_HARDWARE) {
        // Resize renderers to match output resolution
        if (cudaRenderer) cudaRenderer->resize(settings.width, settings.height);
        if (cudaInterop) cudaInterop->resize(settings.width, settings.height);
        if (optixRenderer) optixRenderer->resize(settings.width, settings.height);

        QElapsedTimer frameTimer;
        QElapsedTimer totalTimer;
        totalTimer.start();

        // FFmpeg pipe mode setup
        QProcess* ffmpeg = nullptr;
        QFile* dataFile = nullptr;
        QTextStream* dataStream = nullptr;
        QString videoOutput;
        QString dataFilePath;

        if (settings.use_ffmpeg_pipe) {
            // Start FFmpeg process for direct piping
            QString animName = QDir(outputPath).dirName();
            videoOutput = outputPath + "/" + animName + ".mp4";
            dataFilePath = outputPath + "/" + animName + "_framedata.json";

            ffmpeg = new QProcess();
            QStringList args;
            args << "-y"                              // Overwrite output
                 << "-f" << "rawvideo"                // Raw input format
                 << "-pix_fmt" << "rgba"              // RGBA pixel format
                 << "-s" << QString("%1x%2").arg(settings.width).arg(settings.height)  // Resolution
                 << "-r" << QString::number(static_cast<int>(settings.framerate))  // Input framerate
                 << "-i" << "-"                       // Read from stdin (pipe)
                 << "-vf" << "pad=ceil(iw/2)*2:ceil(ih/2)*2"  // Ensure even dimensions
                 << "-c:v" << "libx264"               // H.264 codec
                 << "-pix_fmt" << "yuv420p"           // Output pixel format
                 << "-crf" << "18"                    // Quality
                 << "-preset" << "fast"               // Encoding speed
                 << videoOutput;

            std::cout << "[FFmpeg Pipe] Starting: ffmpeg " << args.join(" ").toStdString() << std::endl;
            ffmpeg->start("ffmpeg", args);

            if (!ffmpeg->waitForStarted(5000)) {
                std::cerr << "[FFmpeg Pipe] Failed to start ffmpeg" << std::endl;
                delete ffmpeg;
                ffmpeg = nullptr;
                // Fall back to PNG mode
            } else {
                std::cout << "[FFmpeg Pipe] FFmpeg started, piping frames directly" << std::endl;
            }

            // Open data file for frame timing
            dataFile = new QFile(dataFilePath);
            if (dataFile->open(QIODevice::WriteOnly | QIODevice::Text)) {
                dataStream = new QTextStream(dataFile);
                *dataStream << "[\n";  // Start JSON array
            }
        }

        int framesRendered = 0;
        for (int frame = settings.start_frame; frame <= settings.end_frame && batchRendering; ++frame) {
            frameTimer.start();

            int currentFrame = frame - settings.start_frame + 1;
            emit batchRenderProgress(currentFrame, totalFrames);

            // Compute camera position - with or without orbit animation
            vec3 lookfrom;
            if (settings.animate_camera) {
                // Time-based camera animation (matches CPU ray tracer)
                const float framerate = static_cast<float>(settings.framerate);
                const float camera_rotation_speed = 0.4189f;
                float delta_time = 1.0f / framerate;
                float elapsed_time = (frame - settings.start_frame) * delta_time;
                float angle = elapsed_time * camera_rotation_speed;

                // Compute camera position with orbit animation
                float orbitYaw = camYaw + angle;
                float x = camDistance * cosf(camPitch) * sinf(orbitYaw);
                float y = camDistance * sinf(camPitch);
                float z = camDistance * cosf(camPitch) * cosf(orbitYaw);
                lookfrom = camLookAt + vec3(x, y, z);
            } else {
                // Static camera - use current viewport position (no animation)
                float x = camDistance * cosf(camPitch) * sinf(camYaw);
                float y = camDistance * sinf(camPitch);
                float z = camDistance * cosf(camPitch) * cosf(camYaw);
                lookfrom = camLookAt + vec3(x, y, z);
            }

            // Update camera for this frame
            if (settings.render_mode == rt::BatchRenderMode::CUDA_SOFTWARE && cudaRenderer) {
                cudaRenderer->updateCamera(lookfrom, camLookAt, vec3(0, 1, 0), camVFOV, camAperture, camFocusDist);
            } else if (settings.render_mode == rt::BatchRenderMode::OPTIX_HARDWARE && optixRenderer) {
                optixRenderer->updateCamera(lookfrom, camLookAt, vec3(0, 1, 0), camVFOV, camAperture, camFocusDist);
            }

            // Render the frame
            int w = 0, h = 0;
            unsigned char* pixels = nullptr;

            if (settings.render_mode == rt::BatchRenderMode::CUDA_SOFTWARE && cudaRenderer) {
                cudaRenderer->render(settings.samples_per_pixel);
                w = cudaRenderer->getWidth();
                h = cudaRenderer->getHeight();
                vec3* d_fb = cudaRenderer->getFramebuffer();

                if (d_fb && cudaInterop) {
                    pixels = cudaInterop->convertToCPUImage(d_fb, w, h, true);
                }
            } else if (settings.render_mode == rt::BatchRenderMode::OPTIX_HARDWARE && optixRenderer) {
                optixRenderer->render(settings.samples_per_pixel);
                w = optixRenderer->getWidth();
                h = optixRenderer->getHeight();
                float4* d_fb = optixRenderer->getFramebuffer();

                if (d_fb && cudaInterop) {
                    pixels = cudaInterop->convertToCPUImage(d_fb, w, h, true);
                }
            }

            double frameTime = frameTimer.elapsed() / 1000.0;
            double totalElapsed = totalTimer.elapsed() / 1000.0;

            if (pixels) {
                if (ffmpeg && ffmpeg->state() == QProcess::Running) {
                    // Pipe directly to FFmpeg (fast path - no PNG encoding)
                    qint64 bytesToWrite = w * h * 4;
                    qint64 written = ffmpeg->write(reinterpret_cast<char*>(pixels), bytesToWrite);
                    if (written != bytesToWrite) {
                        std::cerr << "[FFmpeg Pipe] Warning: wrote " << written << " of " << bytesToWrite << " bytes" << std::endl;
                    }
                } else {
                    // Fallback: Save frame as PNG
                    QImage image(pixels, w, h, w * 4, QImage::Format_RGBA8888);
                    QString filename = QString("%1/frame_%2.png")
                        .arg(outputPath)
                        .arg(frame, 4, 10, QChar('0'));
                    if (!image.save(filename)) {
                        std::cerr << "[Viewport] Failed to save frame: " << filename.toStdString() << std::endl;
                    }
                }

                // Write frame data to JSON file
                if (dataStream) {
                    if (framesRendered > 0) *dataStream << ",\n";
                    *dataStream << QString("  {\"frame\": %1, \"render_ms\": %2, \"total_elapsed_s\": %3, \"fps\": %4}")
                        .arg(frame)
                        .arg(frameTime * 1000.0, 0, 'f', 2)
                        .arg(totalElapsed, 0, 'f', 3)
                        .arg(1.0 / frameTime, 0, 'f', 1);
                }
            }

            framesRendered++;
            emit batchRenderFrameComplete(frameTime);
        }

        // Cleanup FFmpeg pipe
        if (ffmpeg) {
            ffmpeg->closeWriteChannel();  // Signal end of input
            if (!ffmpeg->waitForFinished(30000)) {
                std::cerr << "[FFmpeg Pipe] FFmpeg encoding timed out" << std::endl;
                ffmpeg->kill();
            } else if (ffmpeg->exitCode() == 0) {
                std::cout << "[FFmpeg Pipe] Video created: " << videoOutput.toStdString() << std::endl;
            } else {
                std::cerr << "[FFmpeg Pipe] FFmpeg error: " << ffmpeg->readAllStandardError().toStdString() << std::endl;
            }
            delete ffmpeg;
        }

        // Close data file
        if (dataStream) {
            *dataStream << "\n]\n";  // End JSON array
            delete dataStream;
        }
        if (dataFile) {
            dataFile->close();
            delete dataFile;
            std::cout << "[Viewport] Frame data saved to: " << dataFilePath.toStdString() << std::endl;
        }

        // Check if we completed all frames or were cancelled
        bool completed = batchRendering;
        batchRendering = false;

        // Restore renderers to viewport resolution
        int vpWidth = width();
        int vpHeight = height();
        if (cudaRenderer) cudaRenderer->resize(vpWidth, vpHeight);
        if (cudaInterop) cudaInterop->resize(vpWidth, vpHeight);
        if (optixRenderer) optixRenderer->resize(vpWidth, vpHeight);

        if (completed) {
            QString message = settings.use_ffmpeg_pipe ?
                QString("Rendered %1 frames to video: %2").arg(totalFrames).arg(videoOutput) :
                QString("Rendered %1 frames to %2").arg(totalFrames).arg(outputPath);
            emit batchRenderComplete(true, message);
        } else {
            emit batchRenderComplete(false, "Batch render cancelled");
        }
    } else {
        // CPU batch rendering
        std::clog << "[Viewport] Starting CPU batch render..." << std::endl;

        // Build CPU scene from sphere primitives
        rebuildCPURayTracingScene(settings.use_bvh);

        if (!cpuScene) {
            std::cerr << "[Viewport] Failed to build CPU scene" << std::endl;
            batchRendering = false;
            emit batchRenderComplete(false, "Failed to build CPU scene");
            return;
        }

        // Set scene on CPU ray tracer
        cpuRayTracer->set_scene(cpuScene);

        // Set lights on CPU ray tracer (convert from CUDA LightData to graphics::Lightf)
        if (cudaRenderer) {
            const auto& cudaLights = cudaRenderer->getLights();
            std::vector<graphics::Lightf> cpuLights;
            cpuLights.reserve(cudaLights.size());

            for (const auto& cl : cudaLights) {
                graphics::Lightf light;
                light.position = Vector3f(cl.position.x(), cl.position.y(), cl.position.z());
                light.direction = Vector3f(cl.direction.x(), cl.direction.y(), cl.direction.z());
                light.color = Color3f(cl.color.x(), cl.color.y(), cl.color.z());
                light.intensity = cl.intensity;
                light.radius = cl.radius;
                // Convert type: 0=point, 1=distant, 2=sphere
                switch (cl.type) {
                    case 0: light.type = graphics::LightType::Point; break;
                    case 1: light.type = graphics::LightType::Distant; break;
                    case 2: light.type = graphics::LightType::Sphere; break;
                    default: light.type = graphics::LightType::Point; break;
                }
                cpuLights.push_back(light);
            }

            cpuRayTracer->set_lights(cpuLights);
            std::clog << "[Viewport] Set " << cpuLights.size() << " lights on CPU ray tracer" << std::endl;
        }

        // Create camera matching current viewport view exactly
        // The Camera class uses spherical coords (r, theta, phi) and compile() computes eye from them.
        // The copy constructor calls compile(), so we MUST set theta/phi correctly.
        //
        // Camera compile() does: eye = SphericalToCartesian(r, theta, phi) then swapYZ()
        // Result: eye = (r*sin(phi)*cos(theta), r*cos(phi), r*sin(phi)*sin(theta))
        //
        // Our getCameraPosition():
        // x = r * cos(pitch) * sin(yaw)
        // y = r * sin(pitch)
        // z = r * cos(pitch) * cos(yaw)
        //
        // Matching: phi = PI/2 - pitch, theta = PI/2 - yaw
        float theta = static_cast<float>(HALF_PI) - camYaw;
        float phi = static_cast<float>(HALF_PI) - camPitch;

        graphics::Cameraf baseCamera(camDistance);
        baseCamera.setPosition(camLookAt.x(), camLookAt.y(), camLookAt.z());  // Sets lookAt
        baseCamera.setRadius(camDistance);
        baseCamera.setRotation(theta, phi);  // Sets theta/phi, calls compile() - eye computed correctly
        baseCamera.setFOV(camVFOV);

        // Run CPU render in background thread
        renderThread = QThread::create([this, settings, baseCamera]() {
            cpuRayTracer->render_animation(
                settings,
                baseCamera,
                [this](int current, int total) {
                    // Queue signal to main thread
                    QMetaObject::invokeMethod(this, [this, current, total]() {
                        emit batchRenderProgress(current, total);
                    }, Qt::QueuedConnection);
                },
                [this](double frameTime) {
                    QMetaObject::invokeMethod(this, [this, frameTime]() {
                        emit batchRenderFrameComplete(frameTime);
                    }, Qt::QueuedConnection);
                },
                [this](bool success, const QString& message) {
                    QMetaObject::invokeMethod(this, [this, success, message]() {
                        batchRendering = false;
                        emit batchRenderComplete(success, message);
                    }, Qt::QueuedConnection);
                }
            );
        });

        renderThread->start();
    }
}

void Viewport::cancelBatchRender() {
    if (cpuRayTracer) {
        cpuRayTracer->request_cancel();
    }
    batchRendering = false;
    std::clog << "[Viewport] Batch render cancelled" << std::endl;
}

void Viewport::rebuildCPURayTracingScene(bool useBVH) {
    std::clog << "[Viewport] rebuildCPURayTracingScene (useBVH=" << useBVH << ")" << std::endl;
    cpuSceneBuilder.clear();

    // Build spherePrimitives from CUDA renderer's sphere data
    if (cudaRenderer) {
        const auto& cudaSpheres = cudaRenderer->getSpheres();
        spherePrimitives.clear();
        spherePrimitives.reserve(cudaSpheres.size());

        for (const auto& cs : cudaSpheres) {
            SphereData sd;
            sd.center = rt::point3(cs.center.x(), cs.center.y(), cs.center.z());
            sd.radius = cs.radius;
            sd.name = cs.name.empty() ? "Sphere" : cs.name;

            // Convert CUDA material to rt::MaterialInfo
            sd.material.albedo = rt::color(cs.albedo.x(), cs.albedo.y(), cs.albedo.z());
            if (cs.material_type == 0) {  // Lambertian
                sd.material.metallic = 0.0;
                sd.material.roughness = 1.0;
                sd.material.is_glass = false;
            } else if (cs.material_type == 1) {  // Metal
                sd.material.metallic = 1.0;
                sd.material.roughness = cs.fuzz_or_ior;
                sd.material.is_glass = false;
            } else if (cs.material_type == 2) {  // Dielectric
                sd.material.metallic = 0.0;
                sd.material.ior = cs.fuzz_or_ior;
                sd.material.is_glass = true;
            }

            spherePrimitives.push_back(sd);
        }
        std::clog << "[Viewport] Copied " << spherePrimitives.size() << " spheres from CUDA renderer" << std::endl;

        // Copy triangles from CUDA renderer
        const auto& cudaTriangles = cudaRenderer->getTriangles();
        std::clog << "[Viewport] Copying " << cudaTriangles.size() << " triangles from CUDA renderer" << std::endl;

        for (const auto& ct : cudaTriangles) {
            // Convert CUDA vec3 to rt::point3/vec3
            rt::point3 v0(ct.v0.x(), ct.v0.y(), ct.v0.z());
            rt::point3 v1(ct.v1.x(), ct.v1.y(), ct.v1.z());
            rt::point3 v2(ct.v2.x(), ct.v2.y(), ct.v2.z());
            rt::vec3 n0(ct.n0.x(), ct.n0.y(), ct.n0.z());
            rt::vec3 n1(ct.n1.x(), ct.n1.y(), ct.n1.z());
            rt::vec3 n2(ct.n2.x(), ct.n2.y(), ct.n2.z());

            // Convert CUDA material to rt::MaterialInfo
            rt::MaterialInfo matInfo;
            matInfo.albedo = rt::color(ct.albedo.x(), ct.albedo.y(), ct.albedo.z());
            if (ct.material_type == 0) {  // Lambertian
                matInfo.metallic = 0.0;
                matInfo.roughness = 1.0;
                matInfo.is_glass = false;
            } else if (ct.material_type == 1) {  // Metal
                matInfo.metallic = 1.0;
                matInfo.roughness = ct.fuzz_or_ior;
                matInfo.is_glass = false;
            } else if (ct.material_type == 2) {  // Dielectric
                matInfo.metallic = 0.0;
                matInfo.ior = ct.fuzz_or_ior;
                matInfo.is_glass = true;
            }

            cpuSceneBuilder.add_triangle(v0, v1, v2, n0, n1, n2, ct.use_vertex_normals, matInfo);
        }
    }

    // Add sphere primitives to the CPU scene
    for (const auto& sphere : spherePrimitives) {
        cpuSceneBuilder.add_sphere(sphere.center, sphere.radius, sphere.material);
    }

    // Build the scene (BVH or list)
    if (useBVH) {
        cpuScene = cpuSceneBuilder.build_bvh();
    } else {
        cpuScene = cpuSceneBuilder.build_list();
    }

    std::clog << "[Viewport] CPU scene built with " << cpuSceneBuilder.get_object_count() << " objects" << std::endl;
}

// ============== Light/Material Modification ==============

int Viewport::getLightCount() const {
    if (cudaRenderer) {
        return static_cast<int>(cudaRenderer->getLights().size());
    }
    return static_cast<int>(lights.size());
}

QStringList Viewport::getObjectNames() const {
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
    if (cudaRenderer) {
        const auto& meshes = cudaRenderer->getMeshes();
        for (size_t i = 0; i < meshes.size(); ++i) {
            if (!meshes[i].name.empty()) {
                names.append(QString::fromStdString(meshes[i].name));
            } else {
                names.append(QString("Mesh %1").arg(i));
            }
        }
    }
    return names;
}

void Viewport::getLightValues(int index, float& r, float& g, float& b, float& intensity) const {
    if (cudaRenderer) {
        const auto& cudaLights = cudaRenderer->getLights();
        if (index >= 0 && index < static_cast<int>(cudaLights.size())) {
            const LightData& light = cudaLights[index];
            r = light.color.x();
            g = light.color.y();
            b = light.color.z();
            intensity = light.intensity;
            return;
        }
    }
    // Fallback to OpenGL lights
    if (index >= 0 && index < static_cast<int>(lights.size())) {
        r = lights[index].color.r();
        g = lights[index].color.g();
        b = lights[index].color.b();
        intensity = lights[index].intensity;
    }
}

void Viewport::getMaterialValues(int index, int& type, float& r, float& g, float& b, float& param) const {
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
    } else if (cudaRenderer) {
        // It's a mesh
        int meshIndex = index - numSpheres;
        const auto& meshes = cudaRenderer->getMeshes();
        if (meshIndex >= 0 && meshIndex < static_cast<int>(meshes.size())) {
            const auto& mesh = meshes[meshIndex];
            type = mesh.material_type;
            r = mesh.albedo.x();
            g = mesh.albedo.y();
            b = mesh.albedo.z();
            param = mesh.fuzz_or_ior;
        }
    }
}

void Viewport::updateLight(int index, float r, float g, float b, float intensity) {
    if (cudaRenderer) {
        auto cudaLights = cudaRenderer->getLights();
        if (index >= 0 && index < static_cast<int>(cudaLights.size())) {
            cudaLights[index].color = vec3(r, g, b);
            cudaLights[index].intensity = intensity;
            cudaRenderer->setLights(cudaLights);

            if (optixRenderer) {
                optixRenderer->setLights(cudaLights);
            }
        }
    }
    // Also update OpenGL lights for that mode
    if (index >= 0 && index < static_cast<int>(lights.size())) {
        lights[index].color = Color3f(r, g, b);
        lights[index].intensity = intensity;
    }
    update();
}

void Viewport::updateMaterial(int objectIndex, int materialType, float r, float g, float b, float param) {
    int numSpheres = static_cast<int>(spherePrimitives.size());
    int numMeshes = cudaRenderer ? static_cast<int>(cudaRenderer->getMeshes().size()) : 0;

    if (objectIndex < 0 || objectIndex >= numSpheres + numMeshes) {
        return;
    }

    if (objectIndex < numSpheres) {
        // Update sphere primitive material
        rt::MaterialInfo& mat = spherePrimitives[objectIndex].material;
        mat.albedo = rt::color(r, g, b);
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

        // Rebuild CUDA materials array for spheres
        if (cudaRenderer) {
            std::vector<MaterialData> cudaMaterials;
            for (const auto& sphere : spherePrimitives) {
                MaterialData m;
                // Convert from roughness/metallic to type
                if (sphere.material.is_glass) {
                    m.type = 2;  // dielectric
                    m.ior = static_cast<float>(sphere.material.ior);
                    m.fuzz = 0.0f;
                } else if (sphere.material.metallic > 0.5) {
                    m.type = 1;  // metal
                    m.fuzz = static_cast<float>(sphere.material.roughness);
                    m.ior = 1.5f;
                } else {
                    m.type = 0;  // lambertian
                    m.fuzz = 0.0f;
                    m.ior = 1.5f;
                }
                m.albedo = vec3(static_cast<float>(sphere.material.albedo.x()),
                               static_cast<float>(sphere.material.albedo.y()),
                               static_cast<float>(sphere.material.albedo.z()));
                cudaMaterials.push_back(m);
            }
            cudaRenderer->setMaterials(cudaMaterials);

            if (optixRenderer) {
                optixRenderer->setMaterials(cudaMaterials);
            }
        }
    } else {
        // Update mesh material
        int meshIndex = objectIndex - numSpheres;
        if (cudaRenderer) {
            float fuzz_or_ior = (materialType == 1) ? param : ((materialType == 2) ? param : 0.0f);
            cudaRenderer->setMeshMaterial(meshIndex, materialType, vec3(r, g, b), fuzz_or_ior);

            if (optixRenderer) {
                optixRenderer->setMeshMaterial(meshIndex, materialType, vec3(r, g, b), fuzz_or_ior);
            }
        }
    }

    // Rebuild CPU scene if needed
    rebuildCPURayTracingScene(true);

    // Restore camera after GPU rebuild (setMaterials/setMeshMaterial recreates GPU camera)
    updateCameraFromOrbit();

    update();
}

}
