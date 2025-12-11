/****************************************************************************
** Meta object code from reading C++ file 'Viewport.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../qt/Viewport.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Viewport.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.10.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN8graphics8ViewportE_t {};
} // unnamed namespace

template <> constexpr inline auto graphics::Viewport::qt_create_metaobjectdata<qt_meta_tag_ZN8graphics8ViewportE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "graphics::Viewport",
        "batchRenderProgress",
        "",
        "currentFrame",
        "totalFrames",
        "batchRenderFrameComplete",
        "frameTimeSeconds",
        "batchRenderComplete",
        "success",
        "message",
        "onTimeout",
        "setSamplesPerPixel",
        "samples",
        "setMaxDepth",
        "depth",
        "getSamplesPerPixel",
        "getMaxDepth",
        "setCudaMaxDepth",
        "getCudaMaxDepth",
        "exportSceneToUSDA",
        "filename",
        "loadSceneFromUSDA",
        "toggleRenderMode",
        "getRenderMode",
        "RenderMode",
        "getRenderModeString",
        "toggleDenoiser",
        "isDenoiserEnabled",
        "loadOBJ",
        "std::string",
        "loadUSD",
        "loadScene",
        "setPendingSceneFile",
        "getCurrentSceneFile",
        "startBatchRender",
        "rt::RenderSettings",
        "settings",
        "cancelBatchRender",
        "isBatchRenderingActive",
        "updateLight",
        "index",
        "r",
        "g",
        "b",
        "intensity",
        "updateMaterial",
        "objectIndex",
        "materialType",
        "param",
        "getLightCount",
        "getObjectNames",
        "getLightValues",
        "float&",
        "getMaterialValues",
        "int&",
        "type"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'batchRenderProgress'
        QtMocHelpers::SignalData<void(int, int)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 3 }, { QMetaType::Int, 4 },
        }}),
        // Signal 'batchRenderFrameComplete'
        QtMocHelpers::SignalData<void(double)>(5, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Double, 6 },
        }}),
        // Signal 'batchRenderComplete'
        QtMocHelpers::SignalData<void(bool, const QString &)>(7, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Bool, 8 }, { QMetaType::QString, 9 },
        }}),
        // Slot 'onTimeout'
        QtMocHelpers::SlotData<void()>(10, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'setSamplesPerPixel'
        QtMocHelpers::SlotData<void(int)>(11, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 12 },
        }}),
        // Slot 'setMaxDepth'
        QtMocHelpers::SlotData<void(int)>(13, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 14 },
        }}),
        // Slot 'getSamplesPerPixel'
        QtMocHelpers::SlotData<int() const>(15, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'getMaxDepth'
        QtMocHelpers::SlotData<int() const>(16, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'setCudaMaxDepth'
        QtMocHelpers::SlotData<void(int)>(17, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 14 },
        }}),
        // Slot 'getCudaMaxDepth'
        QtMocHelpers::SlotData<int() const>(18, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'exportSceneToUSDA'
        QtMocHelpers::SlotData<bool(const QString &)>(19, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { QMetaType::QString, 20 },
        }}),
        // Slot 'loadSceneFromUSDA'
        QtMocHelpers::SlotData<bool(const QString &)>(21, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { QMetaType::QString, 20 },
        }}),
        // Slot 'toggleRenderMode'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'getRenderMode'
        QtMocHelpers::SlotData<RenderMode() const>(23, 2, QMC::AccessPublic, 0x80000000 | 24),
        // Slot 'getRenderModeString'
        QtMocHelpers::SlotData<QString() const>(25, 2, QMC::AccessPublic, QMetaType::QString),
        // Slot 'toggleDenoiser'
        QtMocHelpers::SlotData<void()>(26, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'isDenoiserEnabled'
        QtMocHelpers::SlotData<bool() const>(27, 2, QMC::AccessPublic, QMetaType::Bool),
        // Slot 'loadOBJ'
        QtMocHelpers::SlotData<bool(const std::string &)>(28, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 29, 20 },
        }}),
        // Slot 'loadUSD'
        QtMocHelpers::SlotData<bool(const std::string &)>(30, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 29, 20 },
        }}),
        // Slot 'loadScene'
        QtMocHelpers::SlotData<bool(const std::string &)>(31, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 29, 20 },
        }}),
        // Slot 'setPendingSceneFile'
        QtMocHelpers::SlotData<void(const std::string &)>(32, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 29, 20 },
        }}),
        // Slot 'getCurrentSceneFile'
        QtMocHelpers::SlotData<std::string() const>(33, 2, QMC::AccessPublic, 0x80000000 | 29),
        // Slot 'startBatchRender'
        QtMocHelpers::SlotData<void(const rt::RenderSettings &)>(34, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 35, 36 },
        }}),
        // Slot 'cancelBatchRender'
        QtMocHelpers::SlotData<void()>(37, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'isBatchRenderingActive'
        QtMocHelpers::SlotData<bool() const>(38, 2, QMC::AccessPublic, QMetaType::Bool),
        // Slot 'updateLight'
        QtMocHelpers::SlotData<void(int, float, float, float, float)>(39, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 40 }, { QMetaType::Float, 41 }, { QMetaType::Float, 42 }, { QMetaType::Float, 43 },
            { QMetaType::Float, 44 },
        }}),
        // Slot 'updateMaterial'
        QtMocHelpers::SlotData<void(int, int, float, float, float, float)>(45, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 46 }, { QMetaType::Int, 47 }, { QMetaType::Float, 41 }, { QMetaType::Float, 42 },
            { QMetaType::Float, 43 }, { QMetaType::Float, 48 },
        }}),
        // Slot 'getLightCount'
        QtMocHelpers::SlotData<int() const>(49, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'getObjectNames'
        QtMocHelpers::SlotData<QStringList() const>(50, 2, QMC::AccessPublic, QMetaType::QStringList),
        // Slot 'getLightValues'
        QtMocHelpers::SlotData<void(int, float &, float &, float &, float &) const>(51, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 40 }, { 0x80000000 | 52, 41 }, { 0x80000000 | 52, 42 }, { 0x80000000 | 52, 43 },
            { 0x80000000 | 52, 44 },
        }}),
        // Slot 'getMaterialValues'
        QtMocHelpers::SlotData<void(int, int &, float &, float &, float &, float &) const>(53, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 40 }, { 0x80000000 | 54, 55 }, { 0x80000000 | 52, 41 }, { 0x80000000 | 52, 42 },
            { 0x80000000 | 52, 43 }, { 0x80000000 | 52, 48 },
        }}),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<Viewport, qt_meta_tag_ZN8graphics8ViewportE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject graphics::Viewport::staticMetaObject = { {
    QMetaObject::SuperData::link<QOpenGLWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics8ViewportE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics8ViewportE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN8graphics8ViewportE_t>.metaTypes,
    nullptr
} };

void graphics::Viewport::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<Viewport *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->batchRenderProgress((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2]))); break;
        case 1: _t->batchRenderFrameComplete((*reinterpret_cast<std::add_pointer_t<double>>(_a[1]))); break;
        case 2: _t->batchRenderComplete((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2]))); break;
        case 3: _t->onTimeout(); break;
        case 4: _t->setSamplesPerPixel((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 5: _t->setMaxDepth((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 6: { int _r = _t->getSamplesPerPixel();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 7: { int _r = _t->getMaxDepth();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 8: _t->setCudaMaxDepth((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 9: { int _r = _t->getCudaMaxDepth();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 10: { bool _r = _t->exportSceneToUSDA((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 11: { bool _r = _t->loadSceneFromUSDA((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 12: _t->toggleRenderMode(); break;
        case 13: { RenderMode _r = _t->getRenderMode();
            if (_a[0]) *reinterpret_cast<RenderMode*>(_a[0]) = std::move(_r); }  break;
        case 14: { QString _r = _t->getRenderModeString();
            if (_a[0]) *reinterpret_cast<QString*>(_a[0]) = std::move(_r); }  break;
        case 15: _t->toggleDenoiser(); break;
        case 16: { bool _r = _t->isDenoiserEnabled();
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 17: { bool _r = _t->loadOBJ((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 18: { bool _r = _t->loadUSD((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 19: { bool _r = _t->loadScene((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 20: _t->setPendingSceneFile((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1]))); break;
        case 21: { std::string _r = _t->getCurrentSceneFile();
            if (_a[0]) *reinterpret_cast<std::string*>(_a[0]) = std::move(_r); }  break;
        case 22: _t->startBatchRender((*reinterpret_cast<std::add_pointer_t<rt::RenderSettings>>(_a[1]))); break;
        case 23: _t->cancelBatchRender(); break;
        case 24: { bool _r = _t->isBatchRenderingActive();
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 25: _t->updateLight((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[5]))); break;
        case 26: _t->updateMaterial((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[6]))); break;
        case 27: { int _r = _t->getLightCount();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 28: { QStringList _r = _t->getObjectNames();
            if (_a[0]) *reinterpret_cast<QStringList*>(_a[0]) = std::move(_r); }  break;
        case 29: _t->getLightValues((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[5]))); break;
        case 30: _t->getMaterialValues((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int&>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<float&>>(_a[6]))); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (Viewport::*)(int , int )>(_a, &Viewport::batchRenderProgress, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (Viewport::*)(double )>(_a, &Viewport::batchRenderFrameComplete, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (Viewport::*)(bool , const QString & )>(_a, &Viewport::batchRenderComplete, 2))
            return;
    }
}

const QMetaObject *graphics::Viewport::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *graphics::Viewport::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics8ViewportE_t>.strings))
        return static_cast<void*>(this);
    return QOpenGLWidget::qt_metacast(_clname);
}

int graphics::Viewport::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 31)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 31;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 31)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 31;
    }
    return _id;
}

// SIGNAL 0
void graphics::Viewport::batchRenderProgress(int _t1, int _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2);
}

// SIGNAL 1
void graphics::Viewport::batchRenderFrameComplete(double _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1);
}

// SIGNAL 2
void graphics::Viewport::batchRenderComplete(bool _t1, const QString & _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1, _t2);
}
QT_WARNING_POP
