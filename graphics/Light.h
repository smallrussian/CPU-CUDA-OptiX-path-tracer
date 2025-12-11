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
#ifndef LIGHT_H
#define LIGHT_H

#include "../math/Vector3.h"
#include "Color3.h"

namespace graphics {

// Light types matching CUDA: 0=Point, 1=Distant, 2=Sphere
enum class LightType { Point = 0, Distant = 1, Sphere = 2, Rect = 3 };

/*
 * Light: Represents a light source for ray tracing.
 *
 * Contains position, color, intensity, type, and direction.
 */
template <typename Real>
struct Light {
    Vector3<Real> position;   // World-space position of the light
    Vector3<Real> direction;  // Direction for distant lights
    Color3<Real> color;       // Light color (RGB, typically 0-1 range)
    Real intensity;           // Light intensity multiplier
    Real ambient;             // Ambient light contribution (0-1)
    Real radius;              // Radius for sphere lights
    LightType type;           // Light type (Point, Distant, Sphere)

    // Default constructor - white point light at origin
    Light()
        : position(Vector3<Real>::Zero())
        , direction(Vector3<Real>(Real(0), Real(-1), Real(0)))
        , color(Color3<Real>(Real(1), Real(1), Real(1)))
        , intensity(Real(1))
        , ambient(Real(0.1))
        , radius(Real(1))
        , type(LightType::Point)
    {}

    // Parameterized constructor
    Light(const Vector3<Real>& pos, const Color3<Real>& col, Real intens = Real(1), Real amb = Real(0.1), LightType t = LightType::Point)
        : position(pos)
        , direction(Vector3<Real>(Real(0), Real(-1), Real(0)))
        , color(col)
        , intensity(intens)
        , ambient(amb)
        , radius(Real(1))
        , type(t)
    {}

    // Constructor with position only (white light)
    explicit Light(const Vector3<Real>& pos)
        : position(pos)
        , direction(Vector3<Real>(Real(0), Real(-1), Real(0)))
        , color(Color3<Real>(Real(1), Real(1), Real(1)))
        , intensity(Real(1))
        , ambient(Real(0.1))
        , radius(Real(1))
        , type(LightType::Point)
    {}

    // Copy constructor
    Light(const Light<Real>& other)
        : position(other.position)
        , direction(other.direction)
        , color(other.color)
        , intensity(other.intensity)
        , ambient(other.ambient)
        , radius(other.radius)
        , type(other.type)
    {}

    // Assignment operator
    Light<Real>& operator=(const Light<Real>& other) {
        if (this != &other) {
            position = other.position;
            direction = other.direction;
            color = other.color;
            intensity = other.intensity;
            ambient = other.ambient;
            radius = other.radius;
            type = other.type;
        }
        return *this;
    }

    ~Light() = default;
};

// Convenient typedefs
typedef Light<float> Lightf;
typedef Light<double> Lightd;

}

#endif
