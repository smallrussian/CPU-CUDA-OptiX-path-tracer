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
#ifndef MATHEMATICS_H
#define MATHEMATICS_H

#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Matrix3.h"
#include "Matrix4.h"

namespace graphics {

static const double PI = 3.14159265358979323846;
static const double HALF_PI = PI * 0.5;

/* Convert degrees to radians for real types */
template <typename Real>
inline Real DegreesToRadians(Real degrees) {
    return degrees * PI / Real(180);
}

/* Convert radians to degrees for real types */
template <typename Real>
inline Real RadiansToDegrees(Real radians) {
    return radians * Real(180) / PI;
}

/* R(r, t, p) = rsin(phi)cos(theta)i + rsin(phi)sin(theta)j + rcos(phi)k */
template <typename Real>
inline Vector3<Real> SphericalToCartesian(Real r, Real theta, Real phi) {
    const Real sinPhi = std::sin(phi);
    const Real cosPhi = std::cos(phi);
    const Real sinTheta = std::sin(theta);
    const Real cosTheta = std::cos(theta);
    const Real rs = r * sinPhi;

    return Vector3<Real>(
        rs * cosTheta,
        rs * sinTheta,
        r * cosPhi
    );
}

/* Rt(r, t, p) = -rsin(pphi)sin(theta)i + rsin(phi)cos(theta)j + 0k */
template <typename Real>
inline Vector3<Real> SphericalToCartesian_dTheta(Real r, Real theta, Real phi) {
    const Real sinPhi = std::sin(phi);
    const Real sinTheta = std::sin(theta);
    const Real cosTheta = std::cos(theta);
    const Real rs = r * sinPhi;

    return Vector3<Real>(
        -rs * sinTheta,
        rs * cosTheta,
        Real(0)
    );
}

/* Rp(r, t, p) = rcos(phi)cos(theta)i + rcos(phi)sin(theta)j - rsin(phi)k */
template <typename Real>
inline Vector3<Real> SphericalToCartesian_dPhi(Real r, Real theta, Real phi) {
    const Real cosPhi = std::cos(phi);
    const Real sinPhi = std::sin(phi);
    const Real cosTheta = std::cos(theta);
    const Real sinTheta = std::sin(theta);
    const Real rc = r * cosPhi;

    return Vector3<Real>(
        rc * cosTheta,
        rc * sinTheta,
        -r * sinPhi
    );
}

/* Rp X Rt = r^2 * sin^2(phi)cos(theta)i + r^2 * sin^2(phi)sin(theta)j + r^2 * sin(phi)cos(phi)k */
template <typename Real>
inline Vector3<Real> SphericalToCartesian_dPhiCrossdTheta(Real r, Real theta, Real phi) {
    const Real rs = r * r;
    const Real sinPhi = std::sin(phi);
    const Real cosPhi = std::cos(phi);
    const Real sinTheta = std::sin(theta);
    const Real cosTheta = std::cos(theta);
    const Real sinPhi2 = sinPhi * sinPhi;

    return Vector3<Real>(
        rs * sinPhi2 * cosTheta,
        rs * sinPhi2 * sinTheta,
        rs * sinPhi * cosPhi
    );
}

}

#endif
