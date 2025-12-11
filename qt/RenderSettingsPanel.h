
#ifndef RENDERSETTINGSPANEL_H
#define RENDERSETTINGSPANEL_H

#include "RenderSettingsPanelBase.h"

namespace graphics {

/**
 * Render settings panel for CUDA application.
 * Adds render mode selection (CPU/CUDA/OptiX) on top of base functionality.
 */
class RenderSettingsPanel : public RenderSettingsPanelBase {
    Q_OBJECT

public:
    // Render mode enum (matches Viewport::RenderMode indices for batch rendering)
    enum RenderMode {
        CPU_RAYTRACER = 1,
        CUDA_SOFTWARE = 2,
        OPTIX_HARDWARE = 3
    };

    explicit RenderSettingsPanel(QWidget* parent = nullptr) : RenderSettingsPanelBase(parent) {
        setupUI();
        connectSignals();
    }

    RenderMode getRenderMode() const {
        int index = m_modeComboBox->currentIndex();
        switch (index) {
            case 0: return CPU_RAYTRACER;
            case 1: return CUDA_SOFTWARE;
            case 2: return OPTIX_HARDWARE;
            default: return CUDA_SOFTWARE;
        }
    }

    void setRenderMode(RenderMode mode) {
        int index = 1;
        switch (mode) {
            case CPU_RAYTRACER: index = 0; break;
            case CUDA_SOFTWARE: index = 1; break;
            case OPTIX_HARDWARE: index = 2; break;
        }
        m_modeComboBox->setCurrentIndex(index);
        updateUIForMode(mode);
    }

    void updateUIForMode(RenderMode mode) {
        bool isCPUMode = (mode == CPU_RAYTRACER);

        // Max depth: editable only for CPU mode
        // CUDA uses config.ini, OptiX is compile-time
        m_depthSpinBox->setEnabled(isCPUMode);
        if (mode == CUDA_SOFTWARE) {
            m_depthSpinBox->setToolTip("CUDA max depth is set in config.ini");
        } else if (mode == OPTIX_HARDWARE) {
            m_depthSpinBox->setToolTip("OptiX max depth is compile-time (OptixConfig.h)");
        } else {
            m_depthSpinBox->setToolTip("Maximum ray bounces");
        }

        // BVH and multithreading only for CPU mode
        m_bvhCheckBox->setVisible(isCPUMode);
        m_multithreadCheckBox->setVisible(isCPUMode);

        // GPU video options (Direct to Video) only for GPU modes
        setGpuVideoOptionsEnabled(!isCPUMode);
    }

signals:
    void renderModeChanged(int mode);

private slots:
    void onModeChanged(int index) {
        // Map combo index to RenderMode enum
        RenderMode mode;
        switch (index) {
            case 0: mode = CPU_RAYTRACER; break;
            case 1: mode = CUDA_SOFTWARE; break;
            case 2: mode = OPTIX_HARDWARE; break;
            default: mode = CUDA_SOFTWARE; break;
        }
        updateUIForMode(mode);
        emit renderModeChanged(index);
    }

private:
    void setupUI() {
        QVBoxLayout* mainLayout = new QVBoxLayout(this);
        mainLayout->setSpacing(10);

        // === Render Mode Group (CUDA-specific) ===
        QGroupBox* modeGroup = new QGroupBox("Render Mode");
        QVBoxLayout* modeLayout = new QVBoxLayout(modeGroup);
        m_modeComboBox = new QComboBox();
        m_modeComboBox->blockSignals(true);  // Block signals during setup
        m_modeComboBox->addItem("CPU Ray Tracer (Batch)");
        m_modeComboBox->addItem("CUDA (Real-time)");
        m_modeComboBox->addItem("OptiX (RT Cores)");
        m_modeComboBox->setCurrentIndex(1);  // Default to CUDA
        m_modeComboBox->blockSignals(false);
        m_modeComboBox->setToolTip("Select rendering backend");
        modeLayout->addWidget(m_modeComboBox);
        mainLayout->addWidget(modeGroup);

        // Setup common UI elements from base class
        setupCommonUI(mainLayout);

        // Initialize UI visibility for default mode (CUDA)
        updateUIForMode(CUDA_SOFTWARE);
    }

    void connectSignals() {
        connectCommonSignals();
        connect(m_modeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &RenderSettingsPanel::onModeChanged);
    }

    QComboBox* m_modeComboBox;
};

} // namespace graphics

#endif // RENDERSETTINGSPANEL_H
