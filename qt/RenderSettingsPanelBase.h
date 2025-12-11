
#ifndef RENDERSETTINGSPANELBASE_H
#define RENDERSETTINGSPANELBASE_H

#include <QWidget>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QProgressBar>
#include <QLabel>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QDir>
#include <QCoreApplication>
#include <QProcess>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <iostream>

namespace graphics {

/**
 * Base class for render settings panels.
 * Contains all shared UI elements and functionality for batch rendering.
 * Subclasses can add mode-specific features (e.g., render mode selection for CUDA).
 */
class RenderSettingsPanelBase : public QWidget {
    Q_OBJECT

public:
    explicit RenderSettingsPanelBase(QWidget* parent = nullptr) : QWidget(parent) {}

    // ============== Getters ==============
    int getSamplesPerPixel() const { return m_sppSpinBox->value(); }
    int getMaxDepth() const { return m_depthSpinBox->value(); }
    bool getUseBVH() const { return m_bvhCheckBox->isChecked(); }
    bool getUseMultithreading() const { return m_multithreadCheckBox->isChecked(); }
    int getStartFrame() const { return m_startFrameSpinBox->value(); }
    int getEndFrame() const { return m_endFrameSpinBox->value(); }
    bool getAnimateCamera() const { return m_animateCameraCheckBox->isChecked(); }
    int getOutputWidth() const { return m_widthSpinBox->value(); }
    int getOutputHeight() const { return m_heightSpinBox->value(); }
    QString getOutputPath() const { return m_outputPathEdit->text(); }
    QString getAnimationName() const { return m_animNameEdit->text(); }
    bool getCreateVideo() const { return m_createVideoCheckBox->isChecked(); }
    bool getUseFFmpegPipe() const { return m_ffmpegPipeCheckBox->isChecked(); }
    int getVideoFPS() const { return m_fpsSpinBox->value(); }

    QString getFullOutputPath() const {
        QString basePath = m_outputPathEdit->text();
        QString animName = m_animNameEdit->text().trimmed();
        if (animName.isEmpty()) {
            animName = "untitled";
        }
        return basePath + "/" + animName;
    }

    // ============== Setters ==============
    void setSamplesPerPixel(int value) { m_sppSpinBox->setValue(value); }
    void setMaxDepth(int value) { m_depthSpinBox->setValue(value); }
    void setUseBVH(bool value) { m_bvhCheckBox->setChecked(value); }

    // ============== Light/Material Control Population ==============
    void populateLightSelector(int count) {
        m_lightSelector->clear();
        for (int i = 0; i < count; ++i) {
            m_lightSelector->addItem(QString("Light %1").arg(i));
        }
    }

    void populateMaterialSelector(const QStringList& objectNames) {
        m_materialSelector->clear();
        for (const QString& name : objectNames) {
            m_materialSelector->addItem(name);
        }
    }

    void setLightValues(float r, float g, float b, float intensity) {
        int rVal = static_cast<int>(r * 255);
        int gVal = static_cast<int>(g * 255);
        int bVal = static_cast<int>(b * 255);
        m_lightR->setValue(rVal);
        m_lightG->setValue(gVal);
        m_lightB->setValue(bVal);
        m_lightRLabel->setText(QString::number(rVal));
        m_lightGLabel->setText(QString::number(gVal));
        m_lightBLabel->setText(QString::number(bVal));
        m_lightIntensity->setValue(intensity);
    }

    void setMaterialValues(int type, float r, float g, float b, float param) {
        // Block signals to prevent triggering apply while loading values
        m_materialType->blockSignals(true);
        m_materialType->setCurrentIndex(type);
        m_materialType->blockSignals(false);

        int rVal = static_cast<int>(r * 255);
        int gVal = static_cast<int>(g * 255);
        int bVal = static_cast<int>(b * 255);
        m_matR->setValue(rVal);
        m_matG->setValue(gVal);
        m_matB->setValue(bVal);
        m_matRLabel->setText(QString::number(rVal));
        m_matGLabel->setText(QString::number(gVal));
        m_matBLabel->setText(QString::number(bVal));

        // Update param label and enable state based on type
        onMaterialTypeChanged(type);
        m_matParam->setValue(param);
    }

    QComboBox* lightSelector() const { return m_lightSelector; }
    QComboBox* materialSelector() const { return m_materialSelector; }

    // ============== FFMPEG Video Encoding ==============
    void encodeVideo() {
        QString outputDir = getFullOutputPath();
        QString animName = m_animNameEdit->text().trimmed();
        if (animName.isEmpty()) animName = "untitled";

        QString framePattern = outputDir + "/frame_%04d.png";
        QString videoOutput = outputDir + "/" + animName + ".mp4";
        QString fps = QString::number(m_fpsSpinBox->value());

        QStringList args;
        args << "-y"
             << "-framerate" << fps
             << "-i" << framePattern
             << "-vf" << "pad=ceil(iw/2)*2:ceil(ih/2)*2"
             << "-c:v" << "libx264"
             << "-pix_fmt" << "yuv420p"
             << "-crf" << "18"
             << videoOutput;

        std::cout << "[FFMPEG] Encoding video: " << videoOutput.toStdString() << std::endl;
        std::cout << "[FFMPEG] Command: ffmpeg " << args.join(" ").toStdString() << std::endl;

        QProcess ffmpeg;
        ffmpeg.start("ffmpeg", args);

        if (!ffmpeg.waitForStarted(5000)) {
            std::cerr << "[FFMPEG] Failed to start ffmpeg. Is it installed and in PATH?" << std::endl;
            QMessageBox::warning(this, "Video Encoding Failed",
                "Could not start ffmpeg. Make sure ffmpeg is installed and in your PATH.");
            return;
        }

        if (!ffmpeg.waitForFinished(300000)) {
            std::cerr << "[FFMPEG] Encoding timed out" << std::endl;
            ffmpeg.kill();
            return;
        }

        if (ffmpeg.exitCode() == 0) {
            std::cout << "[FFMPEG] Video created: " << videoOutput.toStdString() << std::endl;
            QMessageBox::information(this, "Video Created",
                QString("Video saved to:\n%1").arg(videoOutput));
        } else {
            QString errorOutput = ffmpeg.readAllStandardError();
            std::cerr << "[FFMPEG] Error: " << errorOutput.toStdString() << std::endl;
            QMessageBox::warning(this, "Video Encoding Failed",
                QString("FFMPEG error:\n%1").arg(errorOutput.left(500)));
        }
    }

    // ============== Progress Updates ==============
    void updateProgress(int currentFrame, int totalFrames) {
        if (totalFrames > 0) {
            int framePercent = (currentFrame * 100) / totalFrames;
            m_progressBar->setValue(framePercent);
            m_progressBar->setFormat(QString("Frame %1/%2 (%p%)").arg(currentFrame).arg(totalFrames));
        }
    }

    void setLastFrameTime(double seconds) {
        m_lastFrameLabel->setText(QString("Last Frame: %1 ms").arg(seconds * 1000.0, 0, 'f', 1));

        m_frameTimeSum += seconds;
        m_frameCount++;
        m_avgFrameTime = m_frameTimeSum / m_frameCount;
    }

    void updateETA(int framesRemaining) {
        if (m_avgFrameTime > 0 && framesRemaining > 0) {
            double totalSeconds = framesRemaining * m_avgFrameTime;
            int minutes = static_cast<int>(totalSeconds) / 60;
            int seconds = static_cast<int>(totalSeconds) % 60;
            m_etaLabel->setText(QString("ETA: ~%1m %2s").arg(minutes).arg(seconds));
        } else {
            m_etaLabel->setText("ETA: calculating...");
        }
    }

    void setETA(const QString& eta) {
        m_etaLabel->setText(QString("ETA: %1").arg(eta));
    }

    void setRenderingState(bool isRendering) {
        m_isRendering = isRendering;
        m_startButton->setEnabled(!isRendering);
        m_cancelButton->setEnabled(isRendering);

        m_sppSpinBox->setEnabled(!isRendering);
        m_depthSpinBox->setEnabled(!isRendering);
        m_bvhCheckBox->setEnabled(!isRendering);
        m_startFrameSpinBox->setEnabled(!isRendering);
        m_endFrameSpinBox->setEnabled(!isRendering);
        m_widthSpinBox->setEnabled(!isRendering);
        m_heightSpinBox->setEnabled(!isRendering);
        m_outputPathEdit->setEnabled(!isRendering);
        m_browseButton->setEnabled(!isRendering);

        if (isRendering) {
            m_frameTimeSum = 0.0;
            m_frameCount = 0;
            m_avgFrameTime = 0.0;
            m_lastFrameLabel->setText("Last Frame: --");
            m_etaLabel->setText("ETA: calculating...");
        } else {
            m_progressBar->setValue(0);
            m_progressBar->setFormat("Frame 0/0 (0%)");
            m_etaLabel->setText("ETA: --");
            m_lastFrameLabel->setText("Last Frame: --");
        }
    }

signals:
    void settingsChanged(int samplesPerPixel, int maxDepth, bool useBVH);
    void startRenderRequested();
    void cancelRenderRequested();
    void lightChanged(int lightIndex, float r, float g, float b, float intensity);
    void materialChanged(int objectIndex, int materialType, float r, float g, float b, float param);

protected slots:
    void onApplyLight() {
        int idx = m_lightSelector->currentIndex();
        float r = m_lightR->value() / 255.0f;
        float g = m_lightG->value() / 255.0f;
        float b = m_lightB->value() / 255.0f;
        float intensity = m_lightIntensity->value();
        emit lightChanged(idx, r, g, b, intensity);
    }

    void onApplyMaterial() {
        int idx = m_materialSelector->currentIndex();
        int type = m_materialType->currentIndex();
        float r = m_matR->value() / 255.0f;
        float g = m_matG->value() / 255.0f;
        float b = m_matB->value() / 255.0f;
        float param = m_matParam->value();
        emit materialChanged(idx, type, r, g, b, param);
    }

    void onMaterialTypeChanged(int type) {
        // Update param label and range based on material type
        if (type == 0) { // Lambertian
            m_matParamLabel->setText("(N/A):");
            m_matParam->setEnabled(false);
        } else if (type == 1) { // Metal
            m_matParamLabel->setText("Fuzz:");
            m_matParam->setRange(0.0, 1.0);
            m_matParam->setValue(0.0);
            m_matParam->setEnabled(true);
        } else if (type == 2) { // Glass
            m_matParamLabel->setText("IOR:");
            m_matParam->setRange(1.0, 3.0);
            m_matParam->setValue(1.5);
            m_matParam->setEnabled(true);
        }
    }

    void onBrowseOutput() {
        QString dir = QFileDialog::getExistingDirectory(this, "Select Output Directory",
                                                         m_outputPathEdit->text());
        if (!dir.isEmpty()) {
            m_outputPathEdit->setText(dir);
        }
    }

    void onOpenOutputFolder() {
        QString outputDir = getFullOutputPath();
        QDir dir(outputDir);

        if (!dir.exists()) {
            outputDir = m_outputPathEdit->text();
            dir.setPath(outputDir);
            if (!dir.exists()) {
                dir.mkpath(outputDir);
            }
        }

        QDesktopServices::openUrl(QUrl::fromLocalFile(outputDir));
    }

    void onSettingsChanged() {
        emit settingsChanged(m_sppSpinBox->value(), m_depthSpinBox->value(), m_bvhCheckBox->isChecked());
    }

    void onStartRender() {
        emit startRenderRequested();
    }

    void onCancelRender() {
        emit cancelRenderRequested();
    }

protected:
    // Setup common UI elements - call from subclass setupUI()
    void setupCommonUI(QVBoxLayout* mainLayout) {
        // === Quality Settings Group ===
        m_qualityGroup = new QGroupBox("Quality Settings");
        QVBoxLayout* qualityLayout = new QVBoxLayout(m_qualityGroup);

        QHBoxLayout* sppLayout = new QHBoxLayout();
        sppLayout->addWidget(new QLabel("Samples/Pixel:"));
        m_sppSpinBox = new QSpinBox();
        m_sppSpinBox->setRange(1, 1000);
        m_sppSpinBox->setValue(10);
        m_sppSpinBox->setToolTip("Higher = less noise, slower render");
        sppLayout->addWidget(m_sppSpinBox);
        qualityLayout->addLayout(sppLayout);

        QHBoxLayout* depthLayout = new QHBoxLayout();
        depthLayout->addWidget(new QLabel("Max Depth:"));
        m_depthSpinBox = new QSpinBox();
        m_depthSpinBox->setRange(1, 100);
        m_depthSpinBox->setValue(10);
        m_depthSpinBox->setToolTip("Maximum ray bounces");
        depthLayout->addWidget(m_depthSpinBox);
        qualityLayout->addLayout(depthLayout);

        m_bvhCheckBox = new QCheckBox("Use BVH Acceleration");
        m_bvhCheckBox->setChecked(true);
        m_bvhCheckBox->setToolTip("Disable to compare BVH vs linear performance");
        qualityLayout->addWidget(m_bvhCheckBox);

        m_multithreadCheckBox = new QCheckBox("Use Multithreading");
        m_multithreadCheckBox->setChecked(true);
        m_multithreadCheckBox->setToolTip("Use all CPU cores for parallel rendering");
        qualityLayout->addWidget(m_multithreadCheckBox);

        mainLayout->addWidget(m_qualityGroup);

        // === Animation Settings Group ===
        m_animGroup = new QGroupBox("Animation");
        QVBoxLayout* animLayout = new QVBoxLayout(m_animGroup);

        m_animateCameraCheckBox = new QCheckBox("Animate Camera (Orbit)");
        m_animateCameraCheckBox->setChecked(true);
        m_animateCameraCheckBox->setToolTip("Uncheck to render a single still frame from current view");
        animLayout->addWidget(m_animateCameraCheckBox);

        QHBoxLayout* frameLayout = new QHBoxLayout();
        frameLayout->addWidget(new QLabel("Frames:"));
        m_startFrameSpinBox = new QSpinBox();
        m_startFrameSpinBox->setRange(1, 9999);
        m_startFrameSpinBox->setValue(1);
        frameLayout->addWidget(m_startFrameSpinBox);
        frameLayout->addWidget(new QLabel("to"));
        m_endFrameSpinBox = new QSpinBox();
        m_endFrameSpinBox->setRange(1, 9999);
        m_endFrameSpinBox->setValue(30);
        frameLayout->addWidget(m_endFrameSpinBox);
        animLayout->addLayout(frameLayout);

        mainLayout->addWidget(m_animGroup);

        // === Output Settings Group ===
        m_outputGroup = new QGroupBox("Output");
        QVBoxLayout* outputLayout = new QVBoxLayout(m_outputGroup);

        QHBoxLayout* resLayout = new QHBoxLayout();
        resLayout->addWidget(new QLabel("Resolution:"));
        m_widthSpinBox = new QSpinBox();
        m_widthSpinBox->setRange(64, 4096);
        m_widthSpinBox->setValue(800);
        resLayout->addWidget(m_widthSpinBox);
        resLayout->addWidget(new QLabel("x"));
        m_heightSpinBox = new QSpinBox();
        m_heightSpinBox->setRange(64, 4096);
        m_heightSpinBox->setValue(600);
        resLayout->addWidget(m_heightSpinBox);
        outputLayout->addLayout(resLayout);

        QHBoxLayout* animNameLayout = new QHBoxLayout();
        animNameLayout->addWidget(new QLabel("Animation:"));
        m_animNameEdit = new QLineEdit();
        m_animNameEdit->setText("anim_001");
        m_animNameEdit->setToolTip("Name for this animation (creates subfolder)");
        m_animNameEdit->setPlaceholderText("e.g., orbit_test, final_render");
        animNameLayout->addWidget(m_animNameEdit);
        outputLayout->addLayout(animNameLayout);

        QHBoxLayout* pathLayout = new QHBoxLayout();
        pathLayout->addWidget(new QLabel("Base Path:"));
        m_outputPathEdit = new QLineEdit();
        QString defaultPath = QCoreApplication::applicationDirPath() + "/../output";
        QDir dir(defaultPath);
        m_outputPathEdit->setText(dir.absolutePath());
        m_outputPathEdit->setToolTip("Base output directory (animation subfolder will be created inside)");
        pathLayout->addWidget(m_outputPathEdit);
        m_browseButton = new QPushButton("...");
        m_browseButton->setFixedWidth(30);
        pathLayout->addWidget(m_browseButton);
        m_openFolderButton = new QPushButton("Open");
        m_openFolderButton->setFixedWidth(45);
        m_openFolderButton->setToolTip("Open output folder in File Explorer");
        pathLayout->addWidget(m_openFolderButton);
        outputLayout->addLayout(pathLayout);

        QHBoxLayout* fpsLayout = new QHBoxLayout();
        fpsLayout->addWidget(new QLabel("Video FPS:"));
        m_fpsSpinBox = new QSpinBox();
        m_fpsSpinBox->setRange(1, 120);
        m_fpsSpinBox->setValue(30);
        m_fpsSpinBox->setToolTip("Framerate for output video (e.g., 30 for 30fps, 60 for 60fps)");
        fpsLayout->addWidget(m_fpsSpinBox);
        fpsLayout->addStretch();
        outputLayout->addLayout(fpsLayout);

        m_ffmpegPipeCheckBox = new QCheckBox("Direct to Video (Fast)");
        m_ffmpegPipeCheckBox->setChecked(true);
        m_ffmpegPipeCheckBox->setToolTip("Pipe frames directly to FFmpeg - faster, no PNG files saved");
        outputLayout->addWidget(m_ffmpegPipeCheckBox);

        m_createVideoCheckBox = new QCheckBox("Create Video After (Slower)");
        m_createVideoCheckBox->setChecked(false);
        m_createVideoCheckBox->setToolTip("Save PNG frames then encode to video - slower but keeps frames");
        outputLayout->addWidget(m_createVideoCheckBox);

        mainLayout->addWidget(m_outputGroup);

        // === Light Controls Group ===
        m_lightGroup = new QGroupBox("Light Controls");
        QVBoxLayout* lightLayout = new QVBoxLayout(m_lightGroup);

        QHBoxLayout* lightSelLayout = new QHBoxLayout();
        lightSelLayout->addWidget(new QLabel("Light:"));
        m_lightSelector = new QComboBox();
        m_lightSelector->addItem("Light 0");
        lightSelLayout->addWidget(m_lightSelector);
        lightLayout->addLayout(lightSelLayout);

        QHBoxLayout* lightRLayout = new QHBoxLayout();
        lightRLayout->addWidget(new QLabel("R:"));
        m_lightR = new QSlider(Qt::Horizontal);
        m_lightR->setRange(0, 255);
        m_lightR->setValue(255);
        m_lightRLabel = new QLabel("255");
        m_lightRLabel->setFixedWidth(30);
        lightRLayout->addWidget(m_lightR);
        lightRLayout->addWidget(m_lightRLabel);
        lightLayout->addLayout(lightRLayout);

        QHBoxLayout* lightGLayout = new QHBoxLayout();
        lightGLayout->addWidget(new QLabel("G:"));
        m_lightG = new QSlider(Qt::Horizontal);
        m_lightG->setRange(0, 255);
        m_lightG->setValue(255);
        m_lightGLabel = new QLabel("255");
        m_lightGLabel->setFixedWidth(30);
        lightGLayout->addWidget(m_lightG);
        lightGLayout->addWidget(m_lightGLabel);
        lightLayout->addLayout(lightGLayout);

        QHBoxLayout* lightBLayout = new QHBoxLayout();
        lightBLayout->addWidget(new QLabel("B:"));
        m_lightB = new QSlider(Qt::Horizontal);
        m_lightB->setRange(0, 255);
        m_lightB->setValue(255);
        m_lightBLabel = new QLabel("255");
        m_lightBLabel->setFixedWidth(30);
        lightBLayout->addWidget(m_lightB);
        lightBLayout->addWidget(m_lightBLabel);
        lightLayout->addLayout(lightBLayout);

        QHBoxLayout* lightIntLayout = new QHBoxLayout();
        lightIntLayout->addWidget(new QLabel("Intensity:"));
        m_lightIntensity = new QDoubleSpinBox();
        m_lightIntensity->setRange(0.0, 100.0);
        m_lightIntensity->setValue(1.0);
        m_lightIntensity->setSingleStep(0.1);
        lightIntLayout->addWidget(m_lightIntensity);
        lightLayout->addLayout(lightIntLayout);

        m_applyLightBtn = new QPushButton("Apply Light");
        lightLayout->addWidget(m_applyLightBtn);

        mainLayout->addWidget(m_lightGroup);

        // === Material Controls Group ===
        m_materialGroup = new QGroupBox("Material Controls");
        QVBoxLayout* matLayout = new QVBoxLayout(m_materialGroup);

        QHBoxLayout* matSelLayout = new QHBoxLayout();
        matSelLayout->addWidget(new QLabel("Object:"));
        m_materialSelector = new QComboBox();
        m_materialSelector->addItem("Sphere 0");
        matSelLayout->addWidget(m_materialSelector);
        matLayout->addLayout(matSelLayout);

        QHBoxLayout* matTypeLayout = new QHBoxLayout();
        matTypeLayout->addWidget(new QLabel("Type:"));
        m_materialType = new QComboBox();
        m_materialType->addItem("Lambertian");
        m_materialType->addItem("Metal");
        m_materialType->addItem("Glass");
        matTypeLayout->addWidget(m_materialType);
        matLayout->addLayout(matTypeLayout);

        QHBoxLayout* matRLayout = new QHBoxLayout();
        matRLayout->addWidget(new QLabel("R:"));
        m_matR = new QSlider(Qt::Horizontal);
        m_matR->setRange(0, 255);
        m_matR->setValue(128);
        m_matRLabel = new QLabel("128");
        m_matRLabel->setFixedWidth(30);
        matRLayout->addWidget(m_matR);
        matRLayout->addWidget(m_matRLabel);
        matLayout->addLayout(matRLayout);

        QHBoxLayout* matGLayout = new QHBoxLayout();
        matGLayout->addWidget(new QLabel("G:"));
        m_matG = new QSlider(Qt::Horizontal);
        m_matG->setRange(0, 255);
        m_matG->setValue(128);
        m_matGLabel = new QLabel("128");
        m_matGLabel->setFixedWidth(30);
        matGLayout->addWidget(m_matG);
        matGLayout->addWidget(m_matGLabel);
        matLayout->addLayout(matGLayout);

        QHBoxLayout* matBLayout = new QHBoxLayout();
        matBLayout->addWidget(new QLabel("B:"));
        m_matB = new QSlider(Qt::Horizontal);
        m_matB->setRange(0, 255);
        m_matB->setValue(128);
        m_matBLabel = new QLabel("128");
        m_matBLabel->setFixedWidth(30);
        matBLayout->addWidget(m_matB);
        matBLayout->addWidget(m_matBLabel);
        matLayout->addLayout(matBLayout);

        QHBoxLayout* matParamLayout = new QHBoxLayout();
        m_matParamLabel = new QLabel("(N/A):");
        matParamLayout->addWidget(m_matParamLabel);
        m_matParam = new QDoubleSpinBox();
        m_matParam->setRange(0.0, 3.0);
        m_matParam->setValue(0.0);
        m_matParam->setSingleStep(0.1);
        m_matParam->setEnabled(false);
        matParamLayout->addWidget(m_matParam);
        matLayout->addLayout(matParamLayout);

        m_applyMaterialBtn = new QPushButton("Apply Material");
        matLayout->addWidget(m_applyMaterialBtn);

        mainLayout->addWidget(m_materialGroup);

        // === Progress Group ===
        m_progressGroup = new QGroupBox("Progress");
        QVBoxLayout* progressLayout = new QVBoxLayout(m_progressGroup);

        m_progressBar = new QProgressBar();
        m_progressBar->setRange(0, 100);
        m_progressBar->setValue(0);
        m_progressBar->setFormat("Frame 0/0 (0%)");
        progressLayout->addWidget(m_progressBar);

        m_lastFrameLabel = new QLabel("Last Frame: --");
        m_lastFrameLabel->setStyleSheet("font-family: monospace;");
        progressLayout->addWidget(m_lastFrameLabel);

        m_etaLabel = new QLabel("ETA: --");
        m_etaLabel->setStyleSheet("font-weight: bold;");
        progressLayout->addWidget(m_etaLabel);

        mainLayout->addWidget(m_progressGroup);

        // === Control Buttons ===
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        m_startButton = new QPushButton("Start Render");
        m_startButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }");
        m_cancelButton = new QPushButton("Cancel");
        m_cancelButton->setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }");
        m_cancelButton->setEnabled(false);
        buttonLayout->addWidget(m_startButton);
        buttonLayout->addWidget(m_cancelButton);
        mainLayout->addLayout(buttonLayout);

        mainLayout->addStretch();
        setMinimumWidth(250);
    }

    void connectCommonSignals() {
        connect(m_browseButton, &QPushButton::clicked, this, &RenderSettingsPanelBase::onBrowseOutput);
        connect(m_openFolderButton, &QPushButton::clicked, this, &RenderSettingsPanelBase::onOpenOutputFolder);
        connect(m_startButton, &QPushButton::clicked, this, &RenderSettingsPanelBase::onStartRender);
        connect(m_cancelButton, &QPushButton::clicked, this, &RenderSettingsPanelBase::onCancelRender);
        connect(m_sppSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderSettingsPanelBase::onSettingsChanged);
        connect(m_depthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderSettingsPanelBase::onSettingsChanged);
        connect(m_bvhCheckBox, &QCheckBox::toggled, this, &RenderSettingsPanelBase::onSettingsChanged);

        // Light and material controls
        connect(m_applyLightBtn, &QPushButton::clicked, this, &RenderSettingsPanelBase::onApplyLight);
        connect(m_applyMaterialBtn, &QPushButton::clicked, this, &RenderSettingsPanelBase::onApplyMaterial);
        connect(m_materialType, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &RenderSettingsPanelBase::onMaterialTypeChanged);

        // Slider-to-label connections for light RGB
        connect(m_lightR, &QSlider::valueChanged, [this](int value) { m_lightRLabel->setText(QString::number(value)); });
        connect(m_lightG, &QSlider::valueChanged, [this](int value) { m_lightGLabel->setText(QString::number(value)); });
        connect(m_lightB, &QSlider::valueChanged, [this](int value) { m_lightBLabel->setText(QString::number(value)); });

        // Slider-to-label connections for material RGB
        connect(m_matR, &QSlider::valueChanged, [this](int value) { m_matRLabel->setText(QString::number(value)); });
        connect(m_matG, &QSlider::valueChanged, [this](int value) { m_matGLabel->setText(QString::number(value)); });
        connect(m_matB, &QSlider::valueChanged, [this](int value) { m_matBLabel->setText(QString::number(value)); });

        // Mutual exclusion for video options - only one can be checked at a time
        connect(m_ffmpegPipeCheckBox, &QCheckBox::toggled, [this](bool checked) {
            if (checked) {
                m_createVideoCheckBox->setChecked(false);
            }
        });
        connect(m_createVideoCheckBox, &QCheckBox::toggled, [this](bool checked) {
            if (checked) {
                m_ffmpegPipeCheckBox->setChecked(false);
            }
        });

        // When animate camera is unchecked, disable frame range and video options (single frame mode)
        connect(m_animateCameraCheckBox, &QCheckBox::toggled, [this](bool checked) {
            m_startFrameSpinBox->setEnabled(checked);
            m_endFrameSpinBox->setEnabled(checked);
            m_fpsSpinBox->setEnabled(checked);
            m_ffmpegPipeCheckBox->setEnabled(checked);
            m_createVideoCheckBox->setEnabled(checked);

            if (!checked) {
                // Single frame mode - save current values and force frame 1 only
                m_savedStartFrame = m_startFrameSpinBox->value();
                m_savedEndFrame = m_endFrameSpinBox->value();
                m_savedFfmpegPipe = m_ffmpegPipeCheckBox->isChecked();
                m_savedCreateVideo = m_createVideoCheckBox->isChecked();

                m_startFrameSpinBox->setValue(1);
                m_endFrameSpinBox->setValue(1);
                m_ffmpegPipeCheckBox->setChecked(false);
                m_createVideoCheckBox->setChecked(false);
            } else {
                // Restore previous frame range and video settings
                m_startFrameSpinBox->setValue(m_savedStartFrame);
                m_endFrameSpinBox->setValue(m_savedEndFrame);
                m_ffmpegPipeCheckBox->setChecked(m_savedFfmpegPipe);
                m_createVideoCheckBox->setChecked(m_savedCreateVideo);
            }
        });
    }

    // Enable/disable video options (for CPU mode where direct piping isn't supported)
    void setGpuVideoOptionsEnabled(bool gpuMode) {
        m_ffmpegPipeCheckBox->setEnabled(gpuMode);
        m_createVideoCheckBox->setEnabled(gpuMode);
        if (!gpuMode) {
            m_ffmpegPipeCheckBox->setChecked(false);
            m_createVideoCheckBox->setChecked(false);
        }
    }

    // UI elements - accessible to subclasses
    QSpinBox* m_sppSpinBox = nullptr;
    QSpinBox* m_depthSpinBox = nullptr;
    QCheckBox* m_bvhCheckBox = nullptr;
    QCheckBox* m_multithreadCheckBox = nullptr;
    QCheckBox* m_animateCameraCheckBox = nullptr;
    QSpinBox* m_startFrameSpinBox = nullptr;
    QSpinBox* m_endFrameSpinBox = nullptr;
    QSpinBox* m_widthSpinBox = nullptr;
    QSpinBox* m_heightSpinBox = nullptr;
    QLineEdit* m_animNameEdit = nullptr;
    QLineEdit* m_outputPathEdit = nullptr;
    QPushButton* m_browseButton = nullptr;
    QPushButton* m_openFolderButton = nullptr;
    QPushButton* m_startButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QProgressBar* m_progressBar = nullptr;
    QLabel* m_lastFrameLabel = nullptr;
    QLabel* m_etaLabel = nullptr;
    QCheckBox* m_createVideoCheckBox = nullptr;
    QCheckBox* m_ffmpegPipeCheckBox = nullptr;
    QSpinBox* m_fpsSpinBox = nullptr;

    // Group boxes for visibility control
    QGroupBox* m_qualityGroup = nullptr;
    QGroupBox* m_animGroup = nullptr;
    QGroupBox* m_outputGroup = nullptr;
    QGroupBox* m_progressGroup = nullptr;
    QGroupBox* m_lightGroup = nullptr;
    QGroupBox* m_materialGroup = nullptr;

    // Light controls
    QComboBox* m_lightSelector = nullptr;
    QSlider* m_lightR = nullptr;
    QSlider* m_lightG = nullptr;
    QSlider* m_lightB = nullptr;
    QLabel* m_lightRLabel = nullptr;
    QLabel* m_lightGLabel = nullptr;
    QLabel* m_lightBLabel = nullptr;
    QDoubleSpinBox* m_lightIntensity = nullptr;
    QPushButton* m_applyLightBtn = nullptr;

    // Material controls
    QComboBox* m_materialSelector = nullptr;
    QComboBox* m_materialType = nullptr;
    QSlider* m_matR = nullptr;
    QSlider* m_matG = nullptr;
    QSlider* m_matB = nullptr;
    QLabel* m_matRLabel = nullptr;
    QLabel* m_matGLabel = nullptr;
    QLabel* m_matBLabel = nullptr;
    QDoubleSpinBox* m_matParam = nullptr;
    QLabel* m_matParamLabel = nullptr;
    QPushButton* m_applyMaterialBtn = nullptr;

    // State
    bool m_isRendering = false;
    double m_frameTimeSum = 0.0;
    int m_frameCount = 0;
    double m_avgFrameTime = 0.0;

    // Saved values for when animate camera is toggled
    int m_savedStartFrame = 1;
    int m_savedEndFrame = 30;
    bool m_savedFfmpegPipe = true;
    bool m_savedCreateVideo = false;
};

} // namespace graphics

#endif // RENDERSETTINGSPANELBASE_H
