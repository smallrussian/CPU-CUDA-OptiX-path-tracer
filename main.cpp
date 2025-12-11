#include <QApplication>
#include <QSurfaceFormat>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSpinBox>
#include <QLabel>
#include <QWidget>
#include <QElapsedTimer>
#include <QTimer>
#include <QFileDialog>
#include <QMessageBox>
#include <QDockWidget>
#include <QFile>
#include <QDir>
#include <QSettings>
#include <QProcess>
#include <QCoreApplication>
#include <iostream>
#include "qt/Viewport.h"
#include "qt/RenderSettingsPanel.h"

// Feature flags
const bool ENABLE_EXPORT_USDA = false;  // Set to true to show Export USDA button

using namespace graphics;
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    // Check for command-line argument (scene file passed from OpenGL version)
    QString pendingSceneFile;
    if (argc > 1) {
        pendingSceneFile = QString::fromLocal8Bit(argv[1]);
        if (!QFile::exists(pendingSceneFile)) {
            std::cerr << "[Main] Scene file not found: " << pendingSceneFile.toStdString() << std::endl;
            pendingSceneFile.clear();
        } else {
            std::cout << "[Main] Scene file from command line: " << pendingSceneFile.toStdString() << std::endl;
        }
    }

    // Show file dialog at startup if no scene file from command line
    if (pendingSceneFile.isEmpty()) {
        pendingSceneFile = QFileDialog::getOpenFileName(
            nullptr,
            "Select Scene to Load",
            "scenes/",
            "Scene Files (*.usda *.usdc *.usd)"
        );

        // Exit if user cancelled (or continue with default RTiOW scene)
        if (pendingSceneFile.isEmpty()) {
            std::cout << "[Main] No scene file selected, using default RTiOW scene" << std::endl;
        } else {
            std::cout << "[Main] Scene file from dialog: " << pendingSceneFile.toStdString() << std::endl;
        }
    }

    // OpenGL 4.1 (needed for CUDA-GL interop)
    QSurfaceFormat format;
    format.setVersion(4, 1);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    // Read config.ini - search up directory tree to find it
    int cudaMaxDepth = 10;
    int defaultSamples = 10;
    QString configPath;

    // Start from current working directory and search up
    QDir searchDir(QDir::currentPath());
    for (int i = 0; i < 5; ++i) {  // Search up to 5 levels
        QString testPath = searchDir.absoluteFilePath("config.ini");
        if (QFile::exists(testPath)) {
            configPath = testPath;
            break;
        }
        if (!searchDir.cdUp()) break;
    }

    if (!configPath.isEmpty()) {
        std::cout << "[Config] Found config.ini at: " << configPath.toStdString() << std::endl;
        QSettings settings(configPath, QSettings::IniFormat);

        cudaMaxDepth = settings.value("CUDA/max_depth", 10).toInt();
        defaultSamples = settings.value("Rendering/default_samples", 10).toInt();

        std::cout << "[Config] CUDA max_depth: " << cudaMaxDepth << std::endl;
        std::cout << "[Config] Default samples: " << defaultSamples << std::endl;
    } else {
        std::cout << "[Config] config.ini not found, using defaults" << std::endl;
    }

    // Create main window
    QMainWindow mainWindow;
    mainWindow.setWindowTitle("CUDA Ray Tracer");
    mainWindow.resize(1200, 850);
    // Create central widget with layout
    QWidget* centralWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(5);
    // Create viewport
    Viewport* viewport = new Viewport();

    // Load scene file if passed from command line (e.g., when switching from OpenGL app)
    if (!pendingSceneFile.isEmpty()) {
        viewport->setPendingSceneFile(pendingSceneFile.toStdString());
    }

    // Create buttons for toolbar
    QPushButton* loadBtn = new QPushButton("Load USDA");
    QPushButton* switchToGLBtn = new QPushButton("Switch to OpenGL");
    loadBtn->setFixedWidth(100);
    switchToGLBtn->setFixedWidth(120);

    // Optional: Export USDA button (hidden by default)
    QPushButton* exportBtn = nullptr;
    if (ENABLE_EXPORT_USDA) {
        exportBtn = new QPushButton("Export USDA");
        exportBtn->setFixedWidth(100);
    }

    // Create horizontal layout for controls (top toolbar)
    QHBoxLayout* controlLayout = new QHBoxLayout();
    controlLayout->addWidget(switchToGLBtn);
    controlLayout->addStretch();
    controlLayout->addWidget(loadBtn);
    if (exportBtn) {
        controlLayout->addWidget(exportBtn);
    }
    // Add widgets to layout
    layout->addWidget(viewport);
    layout->addLayout(controlLayout);
    // Create FPS label overlay
    QLabel* fpsLabel = new QLabel(viewport);
    fpsLabel->setText("FPS: --");
    fpsLabel->setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 150); color: white; padding: 5px; font-family: monospace; }");
    fpsLabel->setFixedSize(100, 25);
    fpsLabel->move(10, 10);
    mainWindow.setCentralWidget(centralWidget);

    // Create and add RenderSettingsPanel as dock widget
    RenderSettingsPanel* settingsPanel = new RenderSettingsPanel();
    QDockWidget* settingsDock = new QDockWidget("Render Settings", &mainWindow);
    settingsDock->setWidget(settingsPanel);
    settingsDock->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    mainWindow.addDockWidget(Qt::RightDockWidgetArea, settingsDock);

    // Apply config.ini values to settings panel
    settingsPanel->setSamplesPerPixel(defaultSamples);
    settingsPanel->setMaxDepth(cudaMaxDepth);

    // Apply CUDA max depth to viewport after GL init (need to use a single-shot timer)
    QTimer::singleShot(100, [viewport, cudaMaxDepth]() {
        viewport->setCudaMaxDepth(cudaMaxDepth);
    });

    // Connect settings panel to viewport
    QObject::connect(settingsPanel, &RenderSettingsPanel::settingsChanged,
        [viewport](int spp, int depth, bool /*useBVH*/) {
            viewport->setSamplesPerPixel(spp);
            viewport->setMaxDepth(depth);
        });

    // Connect batch render signals
    QObject::connect(settingsPanel, &RenderSettingsPanel::startRenderRequested,
        [viewport, settingsPanel, &mainWindow]() {
            // Build render settings from panel
            rt::RenderSettings settings;
            settings.width = settingsPanel->getOutputWidth();
            settings.height = settingsPanel->getOutputHeight();
            settings.samples_per_pixel = settingsPanel->getSamplesPerPixel();
            settings.max_depth = settingsPanel->getMaxDepth();
            settings.use_bvh = settingsPanel->getUseBVH();
            settings.use_multithreading = settingsPanel->getUseMultithreading();
            settings.start_frame = settingsPanel->getStartFrame();
            settings.end_frame = settingsPanel->getEndFrame();
            settings.output_directory = settingsPanel->getFullOutputPath();

            // Set render mode from panel selection
            auto panelMode = settingsPanel->getRenderMode();
            if (panelMode == RenderSettingsPanel::CUDA_SOFTWARE) {
                settings.render_mode = rt::BatchRenderMode::CUDA_SOFTWARE;
            } else if (panelMode == RenderSettingsPanel::OPTIX_HARDWARE) {
                settings.render_mode = rt::BatchRenderMode::OPTIX_HARDWARE;
            } else if (panelMode == RenderSettingsPanel::CPU_RAYTRACER) {
                settings.render_mode = rt::BatchRenderMode::CPU_RAYTRACER;
            }

            // FFmpeg pipe mode
            settings.use_ffmpeg_pipe = settingsPanel->getUseFFmpegPipe();

            // Video framerate
            settings.framerate = settingsPanel->getVideoFPS();

            // Camera animation (uncheck for single still frame)
            settings.animate_camera = settingsPanel->getAnimateCamera();

            settingsPanel->setRenderingState(true);
            viewport->startBatchRender(settings);
        });

    QObject::connect(settingsPanel, &RenderSettingsPanel::cancelRenderRequested,
        [viewport, settingsPanel]() {
            viewport->cancelBatchRender();
            settingsPanel->setRenderingState(false);
        });

    // Connect viewport batch render signals to panel
    QObject::connect(viewport, &Viewport::batchRenderProgress,
        [settingsPanel](int currentFrame, int totalFrames) {
            settingsPanel->updateProgress(currentFrame, totalFrames);
            settingsPanel->updateETA(totalFrames - currentFrame);
        });

    QObject::connect(viewport, &Viewport::batchRenderFrameComplete,
        [settingsPanel](double frameTime) {
            settingsPanel->setLastFrameTime(frameTime);
        });

    QObject::connect(viewport, &Viewport::batchRenderComplete,
        [settingsPanel](bool success, const QString& message) {
            settingsPanel->setRenderingState(false);
            if (success) {
                // Optionally encode video
                if (settingsPanel->getCreateVideo()) {
                    settingsPanel->encodeVideo();
                }
                QMessageBox::information(settingsPanel, "Render Complete", message);
            } else {
                QMessageBox::warning(settingsPanel, "Render", message);
            }
        });

    // Connect render mode changed from panel
    QObject::connect(settingsPanel, &RenderSettingsPanel::renderModeChanged,
        [viewport, &mainWindow](int comboIndex) {
            // Map combo index to RenderMode (0=CPU, 1=CUDA, 2=OptiX)
            // Enum values: CPU_RAYTRACER=1, CUDA_SOFTWARE=2, OPTIX_HARDWARE=3
            RenderMode newMode;
            switch (comboIndex) {
                case 0: newMode = RenderMode::CPU_RAYTRACER; break;
                case 1: newMode = RenderMode::CUDA_SOFTWARE; break;
                case 2: newMode = RenderMode::OPTIX_HARDWARE; break;
                default: newMode = RenderMode::CUDA_SOFTWARE; break;
            }

            if (newMode == RenderMode::CPU_RAYTRACER) {
                // Force preview to CUDA mode (not OptiX)
                if (viewport->getRenderMode() == RenderMode::OPTIX_HARDWARE) {
                    viewport->toggleRenderMode();  // Switch to CUDA
                    mainWindow.setWindowTitle("CUDA Ray Tracer (Software RT)");
                }
                QMessageBox::information(&mainWindow, "CPU Ray Tracer",
                    "CPU Ray Tracer mode selected for batch rendering.\n"
                    "Click 'Start Render' to begin CPU path tracing.\n"
                    "Real-time preview continues using CUDA.");
                return;
            }

            // Switch between CUDA and OptiX
            if (newMode == RenderMode::CUDA_SOFTWARE && viewport->getRenderMode() != RenderMode::CUDA_SOFTWARE) {
                viewport->toggleRenderMode();
                mainWindow.setWindowTitle("CUDA Ray Tracer (Software RT)");
            } else if (newMode == RenderMode::OPTIX_HARDWARE && viewport->getRenderMode() != RenderMode::OPTIX_HARDWARE) {
                viewport->toggleRenderMode();
                mainWindow.setWindowTitle("CUDA Ray Tracer (OptiX RT Cores)");
            }
        });

    // Connect "Switch to OpenGL" button - launches OpenGL app and closes this one
    QObject::connect(switchToGLBtn, &QPushButton::clicked, [viewport, &mainWindow, &a]() {
        QString appDir = QCoreApplication::applicationDirPath();
        QString openglExe = appDir + "/OpenGLRayTracer.exe";

        if (!QFile::exists(openglExe)) {
            QMessageBox::warning(&mainWindow, "Switch to OpenGL",
                QString("OpenGLRayTracer.exe not found in:\n%1\n\n"
                        "Please build the OpenGL project first.").arg(appDir));
            return;
        }

        // Get current scene file to pass to OpenGL app
        QString sceneFile = QString::fromStdString(viewport->getCurrentSceneFile());

        // Launch OpenGL app with scene file as argument (if loaded)
        QStringList args;
        if (!sceneFile.isEmpty()) {
            args << sceneFile;
        }

        if (QProcess::startDetached(openglExe, args)) {
            a.quit();  // Close this app
        } else {
            QMessageBox::warning(&mainWindow, "Switch to OpenGL", "Failed to launch OpenGL app");
        }
    });

    // Connect Export button (if enabled)
    if (exportBtn) {
        QObject::connect(exportBtn, &QPushButton::clicked, [viewport, &mainWindow]() {
            QString filename = QFileDialog::getSaveFileName(&mainWindow,
                "Export Scene", "scenes/rtiow_scene.usda", "USDA Files (*.usda)");
            if (!filename.isEmpty()) {
                if (viewport->exportSceneToUSDA(filename)) {
                    QMessageBox::information(&mainWindow, "Export", "Scene exported successfully!");
                } else {
                    QMessageBox::warning(&mainWindow, "Export", "Failed to export scene.");
                }
            }
        });
    }

    // Connect Load button
    QObject::connect(loadBtn, &QPushButton::clicked, [viewport, settingsPanel, &mainWindow]() {
        QString filename = QFileDialog::getOpenFileName(&mainWindow,
            "Load Scene", "scenes/", "Scene Files (*.usda *.usdc *.usd)");
        if (!filename.isEmpty()) {
            if (viewport->loadSceneFromUSDA(filename)) {
                QMessageBox::information(&mainWindow, "Load", "Scene loaded successfully!");
                // Populate light/material selectors after scene load
                settingsPanel->populateLightSelector(viewport->getLightCount());
                settingsPanel->populateMaterialSelector(viewport->getObjectNames());
            } else {
                QMessageBox::warning(&mainWindow, "Load", "Failed to load scene.");
            }
        }
    });

    // Connect light/material controls
    QObject::connect(settingsPanel, &RenderSettingsPanel::lightChanged,
        [viewport, settingsPanel](int idx, float r, float g, float b, float intensity) {
            viewport->updateLight(idx, r, g, b, intensity);
        });

    QObject::connect(settingsPanel, &RenderSettingsPanel::materialChanged,
        [viewport](int idx, int type, float r, float g, float b, float param) {
            viewport->updateMaterial(idx, type, r, g, b, param);
        });

    // Update UI when light selector changes
    QObject::connect(settingsPanel->lightSelector(), QOverload<int>::of(&QComboBox::currentIndexChanged),
        [viewport, settingsPanel](int idx) {
            if (idx >= 0 && idx < viewport->getLightCount()) {
                float r, g, b, intensity;
                viewport->getLightValues(idx, r, g, b, intensity);
                settingsPanel->setLightValues(r, g, b, intensity);
            }
        });

    // Update UI when material selector changes
    QObject::connect(settingsPanel->materialSelector(), QOverload<int>::of(&QComboBox::currentIndexChanged),
        [viewport, settingsPanel](int idx) {
            int type;
            float r, g, b, param;
            viewport->getMaterialValues(idx, type, r, g, b, param);
            settingsPanel->setMaterialValues(type, r, g, b, param);
        });

    // FPS counter
    QElapsedTimer frameTimer;
    frameTimer.start();
    int frameCount = 0;
    QTimer* fpsTimer = new QTimer();
    fpsTimer->setInterval(500);
    QObject::connect(fpsTimer, &QTimer::timeout, [&frameTimer, &frameCount, fpsLabel]() {
        qint64 elapsed = frameTimer.elapsed();
        if (elapsed > 0) {
            double fps = frameCount * 1000.0 / elapsed;
            fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
        }
        frameCount = 0;
        frameTimer.restart();
    });
    fpsTimer->start();
    // Count frames
    QObject::connect(viewport, &QOpenGLWidget::frameSwapped, [&frameCount]() {
        frameCount++;
    });
    mainWindow.show();

    // Populate light/material selectors after scene is loaded (delayed fallback for scenes without pending file)
    QTimer::singleShot(500, [viewport, settingsPanel]() {
        settingsPanel->populateLightSelector(viewport->getLightCount());
        settingsPanel->populateMaterialSelector(viewport->getObjectNames());
    });

    return a.exec();
}