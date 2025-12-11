# OpenGL Ray Tracer - NO CUDA dependencies
# Uses OpenGL for preview, CPU ray tracer for batch rendering

QT += core gui opengl openglwidgets

CONFIG += c++17
CONFIG -= app_bundle

TEMPLATE = app
TARGET = OpenGL

# ============= Source Files =============
SOURCES += \
    main_opengl.cpp \
    qt/ViewportGL.cpp \
    graphics/Mesh.cpp \
    graphics/Shader.cpp \
    graphics/Texture.cpp

# ============= Header Files =============
HEADERS += \
    qt/ViewportGL.h \
    qt/RenderSettingsPanelBase.h \
    qt/RenderSettingsPanelGL.h \
    math/Mathematics.h \
    math/Matrix3.h \
    math/Matrix4.h \
    math/Quaternion.h \
    math/Transformation.h \
    math/Vector2.h \
    math/Vector3.h \
    math/Vector4.h \
    graphics/Camera.h \
    graphics/Color3.h \
    graphics/Color4.h \
    graphics/Face.h \
    graphics/Light.h \
    graphics/Mesh.h \
    graphics/Shader.h \
    graphics/Texture.h \
    graphics/MouseCamera.h \
    graphics/Vertex.h \
    graphics/Material.h \
    raytracer/rtweekend.h \
    raytracer/vec3.h \
    raytracer/ray.h \
    raytracer/color.h \
    raytracer/interval.h \
    raytracer/hittable.h \
    raytracer/hittable_list.h \
    raytracer/sphere.h \
    raytracer/material.h \
    raytracer/camera.h \
    raytracer/triangle.h \
    raytracer/aabb.h \
    raytracer/bvh.h \
    raytracer/scene_builder.h \
    raytracer/renderer.h \
    usd/USDAParser.h

# Link OpenGL
LIBS += -lopengl32
