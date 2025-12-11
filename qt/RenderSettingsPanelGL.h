
#ifndef RENDERSETTINGSPANELGL_H
#define RENDERSETTINGSPANELGL_H

#include "RenderSettingsPanelBase.h"

namespace graphics {

/**
 * Render settings panel for OpenGL application.
 * Uses CPU ray tracer for batch rendering - shows all quality options.
 * Inherits all common functionality from RenderSettingsPanelBase.
 */
class RenderSettingsPanelGL : public RenderSettingsPanelBase {
    Q_OBJECT

public:
    explicit RenderSettingsPanelGL(QWidget* parent = nullptr) : RenderSettingsPanelBase(parent) {
        setupUI();
        connectSignals();
    }

private:
    void setupUI() {
        QVBoxLayout* mainLayout = new QVBoxLayout(this);
        mainLayout->setSpacing(10);

        // Setup common UI elements from base class
        setupCommonUI(mainLayout);

        // CPU ray tracer always shows BVH and multithreading options
        m_bvhCheckBox->setVisible(true);
        m_multithreadCheckBox->setVisible(true);

        // Disable GPU-only video option (Direct to Video requires GPU)
        setGpuVideoOptionsEnabled(false);
    }

    void connectSignals() {
        connectCommonSignals();
    }
};

} // namespace graphics

#endif // RENDERSETTINGSPANELGL_H
