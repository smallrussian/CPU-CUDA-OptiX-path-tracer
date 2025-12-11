
#ifndef RENDERSETTINGSDIALOG_H
#define RENDERSETTINGSDIALOG_H

#include <QDialog>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>

namespace graphics {

class RenderSettingsDialog : public QDialog {
    Q_OBJECT

public:
    explicit RenderSettingsDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle("Render Settings");
        setFixedSize(300, 200);

        QVBoxLayout* mainLayout = new QVBoxLayout(this);

        // Ray Tracing Settings Group
        QGroupBox* rtGroup = new QGroupBox("Ray Tracing Settings");
        QVBoxLayout* rtLayout = new QVBoxLayout(rtGroup);

        // Samples per pixel
        QHBoxLayout* sppLayout = new QHBoxLayout();
        QLabel* sppLabel = new QLabel("Samples per Pixel:");
        sppSpinBox = new QSpinBox();
        sppSpinBox->setRange(1, 100);
        sppSpinBox->setValue(1);
        sppSpinBox->setToolTip("Higher values = better anti-aliasing but slower.\n1 for real-time, 10-100 for quality.");
        sppLayout->addWidget(sppLabel);
        sppLayout->addStretch();
        sppLayout->addWidget(sppSpinBox);
        rtLayout->addLayout(sppLayout);

        // Max depth
        QHBoxLayout* depthLayout = new QHBoxLayout();
        QLabel* depthLabel = new QLabel("Max Ray Depth:");
        depthSpinBox = new QSpinBox();
        depthSpinBox->setRange(1, 50);
        depthSpinBox->setValue(5);
        depthSpinBox->setToolTip("Maximum ray bounces for reflections/refractions.\n5 is good default, higher for complex scenes.");
        depthLayout->addWidget(depthLabel);
        depthLayout->addStretch();
        depthLayout->addWidget(depthSpinBox);
        rtLayout->addLayout(depthLayout);

        mainLayout->addWidget(rtGroup);
        mainLayout->addStretch();

        // Buttons
        QHBoxLayout* buttonLayout = new QHBoxLayout();
        QPushButton* applyButton = new QPushButton("Apply");
        QPushButton* closeButton = new QPushButton("Close");
        buttonLayout->addStretch();
        buttonLayout->addWidget(applyButton);
        buttonLayout->addWidget(closeButton);
        mainLayout->addLayout(buttonLayout);

        // Connections
        connect(applyButton, &QPushButton::clicked, this, &RenderSettingsDialog::onApply);
        connect(closeButton, &QPushButton::clicked, this, &QDialog::close);
    }

    void setSamplesPerPixel(int value) { sppSpinBox->setValue(value); }
    void setMaxDepth(int value) { depthSpinBox->setValue(value); }
    int getSamplesPerPixel() const { return sppSpinBox->value(); }
    int getMaxDepth() const { return depthSpinBox->value(); }

signals:
    void settingsChanged(int samplesPerPixel, int maxDepth);

private slots:
    void onApply() {
        emit settingsChanged(sppSpinBox->value(), depthSpinBox->value());
    }

private:
    QSpinBox* sppSpinBox;
    QSpinBox* depthSpinBox;
};

} // namespace graphics

#endif
