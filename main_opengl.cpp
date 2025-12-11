/*
 * Copyright (c) Saint Louis University (SLU)
 * Graphics and eXtended Reality (GXR) Laboratory
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <QApplication>
#include <QSurfaceFormat>
#include <QMainWindow>
#include <QDockWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QWidget>
#include <QElapsedTimer>
#include <QTimer>
#include <QFileDialog>
#include <QFileInfo>
#include <QProcess>
#include <QScreen>
#include <QFile>
#include <QCoreApplication>
#include <QMessageBox>
#include "qt/ViewportGL.h"
#include "qt/RenderSettingsPanelGL.h"

using namespace graphics;

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    QString filename;

    // Check for command-line argument (scene file passed from CUDA version)
    if (argc > 1) {
        filename = QString::fromLocal8Bit(argv[1]);
        if (!QFile::exists(filename)) {
            filename.clear();  // Invalid file, show dialog
        }
    }

    // Show file dialog if no valid file from command line
    if (filename.isEmpty()) {
        filename = QFileDialog::getOpenFileName(
            nullptr,
            "Select Scene to Load",
            ".",
            "Scene Files (*.usda *.usdc *.usd)"
        );

        // Exit if user cancelled
        if (filename.isEmpty()) {
            return 0;
        }
    }

    // OpenGL 4.1
    QSurfaceFormat format;
    format.setVersion(4, 1);
    format.setProfile(QSurfaceFormat::CoreProfile);
    QSurfaceFormat::setDefaultFormat(format);

    // Create main window
    QMainWindow mainWindow;
    mainWindow.setWindowTitle(QString("Phong Shading [OpenGL] - %1").arg(QFileInfo(filename).fileName()));
    mainWindow.resize(1100, 700);  // Wider to accommodate dock panel


    // Create central widget with layout
    QWidget* centralWidget = new QWidget();
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(5);

    // Create viewport and set pending scene file
    ViewportGL* viewport = new ViewportGL();
    viewport->setPendingSceneFile(filename.toStdString());

    // Create load scene button
    QPushButton* loadButton = new QPushButton("Load Scene...");
    loadButton->setFixedHeight(30);

    // Create switch to CUDA button
    QPushButton* switchToCUDABtn = new QPushButton("Switch to CUDA/OptiX");
    switchToCUDABtn->setFixedHeight(30);
    switchToCUDABtn->setFixedWidth(140);

    // Create horizontal layout for buttons
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    buttonLayout->addWidget(loadButton);
    buttonLayout->addWidget(switchToCUDABtn);
    buttonLayout->addStretch();

    // Add widgets to layout
    layout->addWidget(viewport);
    layout->addLayout(buttonLayout);

    // Create FPS label overlay (parented to viewport, positioned in top-left)
    QLabel* fpsLabel = new QLabel(viewport);
    fpsLabel->setText("FPS: --");
    fpsLabel->setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 150); color: white; padding: 5px; font-family: monospace; }");
    fpsLabel->setFixedSize(130, 25);
    fpsLabel->move(10, 10);

    mainWindow.setCentralWidget(centralWidget);

    // === Create Docked Render Settings Panel ===
    QDockWidget* settingsDock = new QDockWidget("Render Settings", &mainWindow);
    settingsDock->setAllowedAreas(Qt::RightDockWidgetArea | Qt::LeftDockWidgetArea);
    settingsDock->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);

    RenderSettingsPanelGL* settingsPanel = new RenderSettingsPanelGL();
    settingsPanel->setSamplesPerPixel(viewport->getSamplesPerPixel());
    settingsPanel->setMaxDepth(viewport->getMaxDepth());
    settingsDock->setWidget(settingsPanel);

    mainWindow.addDockWidget(Qt::RightDockWidgetArea, settingsDock);

    // Connect load button - restart app with new scene to ensure clean initialization
    QObject::connect(loadButton, &QPushButton::clicked, [&mainWindow, &a]() {
        QString filename = QFileDialog::getOpenFileName(
            &mainWindow,
            "Load Scene",
            ".",
            "Scene Files (*.usda *.usdc *.usd *.obj);;USD Files (*.usda *.usdc *.usd);;OBJ Files (*.obj)"
        );

        if (!filename.isEmpty()) {
            // Restart the application with the new scene file as argument
            QString program = QCoreApplication::applicationFilePath();
            QStringList arguments;
            arguments << filename;
            QProcess::startDetached(program, arguments);
            a.quit();
        }
    });

    // Connect settings panel signals
    QObject::connect(settingsPanel, &RenderSettingsPanelGL::settingsChanged,
        [viewport](int spp, int depth, bool useBVH) {
            viewport->setSamplesPerPixel(spp);
            viewport->setMaxDepth(depth);
            // TODO: Connect BVH toggle when implemented
            (void)useBVH;  // Suppress unused warning for now
        });

    // Connect start render button
    QObject::connect(settingsPanel, &RenderSettingsPanelGL::startRenderRequested,
        [viewport, settingsPanel]() {
            RenderSettings settings;
            settings.width = settingsPanel->getOutputWidth();
            settings.height = settingsPanel->getOutputHeight();
            settings.samples_per_pixel = settingsPanel->getSamplesPerPixel();
            settings.max_depth = settingsPanel->getMaxDepth();
            settings.use_bvh = settingsPanel->getUseBVH();
            settings.use_multithreading = settingsPanel->getUseMultithreading();
            settings.start_frame = settingsPanel->getStartFrame();
            settings.end_frame = settingsPanel->getEndFrame();
            settings.output_directory = settingsPanel->getFullOutputPath();  // Includes animation subfolder
            settings.animate_camera = settingsPanel->getAnimateCamera();

            settingsPanel->setRenderingState(true);
            viewport->startBatchRender(settings);
        });

    // Connect cancel render button
    QObject::connect(settingsPanel, &RenderSettingsPanelGL::cancelRenderRequested,
        [viewport]() {
            viewport->cancelBatchRender();
        });

    // Connect light/material controls
    QObject::connect(settingsPanel, &RenderSettingsPanelGL::lightChanged,
        [viewport](int idx, float r, float g, float b, float intensity) {
            viewport->updateLight(idx, r, g, b, intensity);
        });

    QObject::connect(settingsPanel, &RenderSettingsPanelGL::materialChanged,
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

    // Connect progress signal from viewport to settings panel (frame-level only)
    QObject::connect(viewport, &ViewportGL::batchRenderProgress,
        [settingsPanel](int currentFrame, int totalFrames) {
            settingsPanel->updateProgress(currentFrame, totalFrames);
            // Update ETA based on frames remaining
            int framesRemaining = totalFrames - currentFrame;
            settingsPanel->updateETA(framesRemaining);
        });

    // Connect frame complete signal for timing
    QObject::connect(viewport, &ViewportGL::batchRenderFrameComplete,
        settingsPanel, &RenderSettingsPanelGL::setLastFrameTime);

    // Connect completion signal
    QObject::connect(viewport, &ViewportGL::batchRenderComplete,
        [settingsPanel](bool success, const QString& message) {
            settingsPanel->setRenderingState(false);
            if (success) {
                std::cout << "[Main] " << message.toStdString() << std::endl;

                // Encode video if option is enabled
                if (settingsPanel->getCreateVideo()) {
                    std::cout << "[Main] Starting video encoding..." << std::endl;
                    settingsPanel->encodeVideo();
                }
            } else {
                std::cerr << "[Main] " << message.toStdString() << std::endl;
            }
        });

    // Connect Switch to CUDA/OptiX button
    QObject::connect(switchToCUDABtn, &QPushButton::clicked, [viewport, &mainWindow, &a]() {
        // Get current scene file
        QString sceneFile = QString::fromStdString(viewport->getCurrentSceneFile());

        // Check if it's an OBJ file - CUDA doesn't support OBJ yet
        if (sceneFile.toLower().endsWith(".obj")) {
            QMessageBox::warning(&mainWindow, "OBJ Not Supported",
                "The CUDA renderer does not support OBJ files yet.\n\n"
                "Please load a USDA scene file to use CUDA/OptiX rendering.");
            return;
        }

        // Find CUDA executable (same directory as current exe)
        QString appDir = QCoreApplication::applicationDirPath();
        QString cudaExe = appDir + "/FinalProject.exe";

        if (!QFile::exists(cudaExe)) {
            QMessageBox::warning(&mainWindow, "Error",
                "PhongShading.exe not found in:\n" + appDir +
                "\n\nPlease build the CUDA version first.");
            return;
        }

        // Launch CUDA version with scene file as argument
        QStringList args;
        if (!sceneFile.isEmpty()) {
            args << sceneFile;
        }

        if (QProcess::startDetached(cudaExe, args)) {
            a.quit();  // Close this application
        } else {
            QMessageBox::warning(&mainWindow, "Error", "Failed to launch CUDA Ray Tracer");
        }
    });

    // FPS counter
    QElapsedTimer frameTimer;
    frameTimer.start();
    int frameCount = 0;

    QTimer* fpsTimer = new QTimer();
    fpsTimer->setInterval(500);  // Update FPS display every 500ms
    QObject::connect(fpsTimer, &QTimer::timeout, [&frameTimer, &frameCount, fpsLabel]() {
        qint64 elapsed = frameTimer.elapsed();
        if (elapsed > 0) {
            double fps = frameCount * 1000.0 / elapsed;
            fpsLabel->setText(QString("FPS: %1 [GL]").arg(fps, 0, 'f', 1));
        }
        frameCount = 0;
        frameTimer.restart();
    });
    fpsTimer->start();

    // Count frames on each paint
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
