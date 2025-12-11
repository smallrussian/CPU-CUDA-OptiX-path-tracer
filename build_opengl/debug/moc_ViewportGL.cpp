/****************************************************************************
** Meta object code from reading C++ file 'ViewportGL.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../qt/ViewportGL.h"
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ViewportGL.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN8graphics10ViewportGLE_t {};
} // unnamed namespace

template <> constexpr inline auto graphics::ViewportGL::qt_create_metaobjectdata<qt_meta_tag_ZN8graphics10ViewportGLE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "graphics::ViewportGL",
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
        "toggleRayTracing",
        "isRayTracingEnabled",
        "loadOBJ",
        "std::string",
        "filename",
        "loadUSD",
        "loadScene",
        "setPendingSceneFile",
        "setSamplesPerPixel",
        "samples",
        "setMaxDepth",
        "depth",
        "getSamplesPerPixel",
        "getMaxDepth",
        "startBatchRender",
        "RenderSettings",
        "settings",
        "cancelBatchRender",
        "isBatchRenderingActive",
        "getCurrentSceneFile"
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
        // Slot 'toggleRayTracing'
        QtMocHelpers::SlotData<void()>(11, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'isRayTracingEnabled'
        QtMocHelpers::SlotData<bool() const>(12, 2, QMC::AccessPublic, QMetaType::Bool),
        // Slot 'loadOBJ'
        QtMocHelpers::SlotData<bool(const std::string &)>(13, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 14, 15 },
        }}),
        // Slot 'loadUSD'
        QtMocHelpers::SlotData<bool(const std::string &)>(16, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 14, 15 },
        }}),
        // Slot 'loadScene'
        QtMocHelpers::SlotData<bool(const std::string &)>(17, 2, QMC::AccessPublic, QMetaType::Bool, {{
            { 0x80000000 | 14, 15 },
        }}),
        // Slot 'setPendingSceneFile'
        QtMocHelpers::SlotData<void(const std::string &)>(18, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 14, 15 },
        }}),
        // Slot 'setSamplesPerPixel'
        QtMocHelpers::SlotData<void(int)>(19, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 20 },
        }}),
        // Slot 'setMaxDepth'
        QtMocHelpers::SlotData<void(int)>(21, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 22 },
        }}),
        // Slot 'getSamplesPerPixel'
        QtMocHelpers::SlotData<int() const>(23, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'getMaxDepth'
        QtMocHelpers::SlotData<int() const>(24, 2, QMC::AccessPublic, QMetaType::Int),
        // Slot 'startBatchRender'
        QtMocHelpers::SlotData<void(const RenderSettings &)>(25, 2, QMC::AccessPublic, QMetaType::Void, {{
            { 0x80000000 | 26, 27 },
        }}),
        // Slot 'cancelBatchRender'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPublic, QMetaType::Void),
        // Slot 'isBatchRenderingActive'
        QtMocHelpers::SlotData<bool() const>(29, 2, QMC::AccessPublic, QMetaType::Bool),
        // Slot 'getCurrentSceneFile'
        QtMocHelpers::SlotData<std::string() const>(30, 2, QMC::AccessPublic, 0x80000000 | 14),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<ViewportGL, qt_meta_tag_ZN8graphics10ViewportGLE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject graphics::ViewportGL::staticMetaObject = { {
    QMetaObject::SuperData::link<QOpenGLWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics10ViewportGLE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics10ViewportGLE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN8graphics10ViewportGLE_t>.metaTypes,
    nullptr
} };

void graphics::ViewportGL::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<ViewportGL *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->batchRenderProgress((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2]))); break;
        case 1: _t->batchRenderFrameComplete((*reinterpret_cast<std::add_pointer_t<double>>(_a[1]))); break;
        case 2: _t->batchRenderComplete((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2]))); break;
        case 3: _t->onTimeout(); break;
        case 4: _t->toggleRayTracing(); break;
        case 5: { bool _r = _t->isRayTracingEnabled();
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 6: { bool _r = _t->loadOBJ((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 7: { bool _r = _t->loadUSD((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 8: { bool _r = _t->loadScene((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1])));
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 9: _t->setPendingSceneFile((*reinterpret_cast<std::add_pointer_t<std::string>>(_a[1]))); break;
        case 10: _t->setSamplesPerPixel((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 11: _t->setMaxDepth((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 12: { int _r = _t->getSamplesPerPixel();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 13: { int _r = _t->getMaxDepth();
            if (_a[0]) *reinterpret_cast<int*>(_a[0]) = std::move(_r); }  break;
        case 14: _t->startBatchRender((*reinterpret_cast<std::add_pointer_t<RenderSettings>>(_a[1]))); break;
        case 15: _t->cancelBatchRender(); break;
        case 16: { bool _r = _t->isBatchRenderingActive();
            if (_a[0]) *reinterpret_cast<bool*>(_a[0]) = std::move(_r); }  break;
        case 17: { std::string _r = _t->getCurrentSceneFile();
            if (_a[0]) *reinterpret_cast<std::string*>(_a[0]) = std::move(_r); }  break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (ViewportGL::*)(int , int )>(_a, &ViewportGL::batchRenderProgress, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (ViewportGL::*)(double )>(_a, &ViewportGL::batchRenderFrameComplete, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (ViewportGL::*)(bool , const QString & )>(_a, &ViewportGL::batchRenderComplete, 2))
            return;
    }
}

const QMetaObject *graphics::ViewportGL::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *graphics::ViewportGL::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics10ViewportGLE_t>.strings))
        return static_cast<void*>(this);
    return QOpenGLWidget::qt_metacast(_clname);
}

int graphics::ViewportGL::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QOpenGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 18)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 18;
    }
    return _id;
}

// SIGNAL 0
void graphics::ViewportGL::batchRenderProgress(int _t1, int _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2);
}

// SIGNAL 1
void graphics::ViewportGL::batchRenderFrameComplete(double _t1)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 1, nullptr, _t1);
}

// SIGNAL 2
void graphics::ViewportGL::batchRenderComplete(bool _t1, const QString & _t2)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 2, nullptr, _t1, _t2);
}
QT_WARNING_POP
