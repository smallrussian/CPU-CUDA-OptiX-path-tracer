/****************************************************************************
** Meta object code from reading C++ file 'RenderSettingsPanelBase.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.10.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../qt/RenderSettingsPanelBase.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RenderSettingsPanelBase.h' doesn't include <QObject>."
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
struct qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t {};
} // unnamed namespace

template <> constexpr inline auto graphics::RenderSettingsPanelBase::qt_create_metaobjectdata<qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "graphics::RenderSettingsPanelBase",
        "settingsChanged",
        "",
        "samplesPerPixel",
        "maxDepth",
        "useBVH",
        "startRenderRequested",
        "cancelRenderRequested",
        "lightChanged",
        "lightIndex",
        "r",
        "g",
        "b",
        "intensity",
        "materialChanged",
        "objectIndex",
        "materialType",
        "param",
        "onApplyLight",
        "onApplyMaterial",
        "onMaterialTypeChanged",
        "type",
        "onBrowseOutput",
        "onOpenOutputFolder",
        "onSettingsChanged",
        "onStartRender",
        "onCancelRender"
    };

    QtMocHelpers::UintData qt_methods {
        // Signal 'settingsChanged'
        QtMocHelpers::SignalData<void(int, int, bool)>(1, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 3 }, { QMetaType::Int, 4 }, { QMetaType::Bool, 5 },
        }}),
        // Signal 'startRenderRequested'
        QtMocHelpers::SignalData<void()>(6, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'cancelRenderRequested'
        QtMocHelpers::SignalData<void()>(7, 2, QMC::AccessPublic, QMetaType::Void),
        // Signal 'lightChanged'
        QtMocHelpers::SignalData<void(int, float, float, float, float)>(8, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 9 }, { QMetaType::Float, 10 }, { QMetaType::Float, 11 }, { QMetaType::Float, 12 },
            { QMetaType::Float, 13 },
        }}),
        // Signal 'materialChanged'
        QtMocHelpers::SignalData<void(int, int, float, float, float, float)>(14, 2, QMC::AccessPublic, QMetaType::Void, {{
            { QMetaType::Int, 15 }, { QMetaType::Int, 16 }, { QMetaType::Float, 10 }, { QMetaType::Float, 11 },
            { QMetaType::Float, 12 }, { QMetaType::Float, 17 },
        }}),
        // Slot 'onApplyLight'
        QtMocHelpers::SlotData<void()>(18, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onApplyMaterial'
        QtMocHelpers::SlotData<void()>(19, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onMaterialTypeChanged'
        QtMocHelpers::SlotData<void(int)>(20, 2, QMC::AccessProtected, QMetaType::Void, {{
            { QMetaType::Int, 21 },
        }}),
        // Slot 'onBrowseOutput'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onOpenOutputFolder'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onSettingsChanged'
        QtMocHelpers::SlotData<void()>(24, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onStartRender'
        QtMocHelpers::SlotData<void()>(25, 2, QMC::AccessProtected, QMetaType::Void),
        // Slot 'onCancelRender'
        QtMocHelpers::SlotData<void()>(26, 2, QMC::AccessProtected, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<RenderSettingsPanelBase, qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject graphics::RenderSettingsPanelBase::staticMetaObject = { {
    QMetaObject::SuperData::link<QWidget::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>.metaTypes,
    nullptr
} };

void graphics::RenderSettingsPanelBase::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<RenderSettingsPanelBase *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->settingsChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[3]))); break;
        case 1: _t->startRenderRequested(); break;
        case 2: _t->cancelRenderRequested(); break;
        case 3: _t->lightChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[5]))); break;
        case 4: _t->materialChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<int>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[4])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[5])),(*reinterpret_cast<std::add_pointer_t<float>>(_a[6]))); break;
        case 5: _t->onApplyLight(); break;
        case 6: _t->onApplyMaterial(); break;
        case 7: _t->onMaterialTypeChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 8: _t->onBrowseOutput(); break;
        case 9: _t->onOpenOutputFolder(); break;
        case 10: _t->onSettingsChanged(); break;
        case 11: _t->onStartRender(); break;
        case 12: _t->onCancelRender(); break;
        default: ;
        }
    }
    if (_c == QMetaObject::IndexOfMethod) {
        if (QtMocHelpers::indexOfMethod<void (RenderSettingsPanelBase::*)(int , int , bool )>(_a, &RenderSettingsPanelBase::settingsChanged, 0))
            return;
        if (QtMocHelpers::indexOfMethod<void (RenderSettingsPanelBase::*)()>(_a, &RenderSettingsPanelBase::startRenderRequested, 1))
            return;
        if (QtMocHelpers::indexOfMethod<void (RenderSettingsPanelBase::*)()>(_a, &RenderSettingsPanelBase::cancelRenderRequested, 2))
            return;
        if (QtMocHelpers::indexOfMethod<void (RenderSettingsPanelBase::*)(int , float , float , float , float )>(_a, &RenderSettingsPanelBase::lightChanged, 3))
            return;
        if (QtMocHelpers::indexOfMethod<void (RenderSettingsPanelBase::*)(int , int , float , float , float , float )>(_a, &RenderSettingsPanelBase::materialChanged, 4))
            return;
    }
}

const QMetaObject *graphics::RenderSettingsPanelBase::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *graphics::RenderSettingsPanelBase::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN8graphics23RenderSettingsPanelBaseE_t>.strings))
        return static_cast<void*>(this);
    return QWidget::qt_metacast(_clname);
}

int graphics::RenderSettingsPanelBase::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 13;
    }
    return _id;
}

// SIGNAL 0
void graphics::RenderSettingsPanelBase::settingsChanged(int _t1, int _t2, bool _t3)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 0, nullptr, _t1, _t2, _t3);
}

// SIGNAL 1
void graphics::RenderSettingsPanelBase::startRenderRequested()
{
    QMetaObject::activate(this, &staticMetaObject, 1, nullptr);
}

// SIGNAL 2
void graphics::RenderSettingsPanelBase::cancelRenderRequested()
{
    QMetaObject::activate(this, &staticMetaObject, 2, nullptr);
}

// SIGNAL 3
void graphics::RenderSettingsPanelBase::lightChanged(int _t1, float _t2, float _t3, float _t4, float _t5)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 3, nullptr, _t1, _t2, _t3, _t4, _t5);
}

// SIGNAL 4
void graphics::RenderSettingsPanelBase::materialChanged(int _t1, int _t2, float _t3, float _t4, float _t5, float _t6)
{
    QMetaObject::activate<void>(this, &staticMetaObject, 4, nullptr, _t1, _t2, _t3, _t4, _t5, _t6);
}
QT_WARNING_POP
