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
#ifndef QUATERNION_H
#define QUATERNION_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <iomanip>
#include <stdexcept>

#include "Vector3.h"
#include "Vector4.h"
#include "Matrix3.h"

template <typename Real>
using RotationMatrix = graphics::Matrix3<Real>;

namespace graphics {

template <typename Real>
class Quaternion;

template <typename Real>
std::ostream& operator << (std::ostream& out, const Quaternion<Real>& q);

/* * Quaternion: Representation of a 4x1 quaternion.
 *
 * Mathematical Definition:
 * q = w + xi + yj + zk
 *
 * Memory Definition (x, y, z, w):
 * [data[0], data[1], data[2], data[3]]
 */
template <typename Real>
class Quaternion {
    static_assert(std::is_floating_point<Real>::value, 
    "[Quaternion:Type] Error: Quaternion Real type must be a floating point numerical representation.");

    enum Element { X, Y, Z, W, COMPONENT_COUNT};

public:
    Quaternion(bool identity = true);
    Quaternion(const Quaternion<Real>& q);
    Quaternion(const Vector3<Real>& axis, Real angle);
    Quaternion(Real x, Real y, Real z, Real w);
    Quaternion(Real yaw, Real pitch, Real roll);
    Quaternion(const RotationMatrix<Real>& rotationMatrix);
    virtual ~Quaternion();

    void normalize();
    void identity();
    bool isIdentity() const;
    void multiply(const Quaternion<Real>& q);
    void multiplyOnLeft(const Quaternion<Real>& q);
    void multiplyOnRight(const Quaternion<Real>& q);
    Quaternion<Real> normalized() const;
    RotationMatrix<Real> toRotationMatrix() const;
    Vector4<Real> toVector() const;
    Real length() const;

    void fromVector(const Vector4<Real>& v);
    void fromAxisAngle(const Vector3<Real>& axis, Real angle);
    void fromEulerAngles(Real yaw, Real pitch, Real roll);
    void fromRotationMatrix(const RotationMatrix<Real>& rotationMatrix);
    void fromEulerRotationX(Real angle);
    void fromEulerRotationY(Real angle);
    void fromEulerRotationZ(Real angle);

    void set(Real x, Real y, Real z, Real w);
    void set(const Vector4<Real>& v);
    void setX(Real x);
    void setY(Real y);
    void setZ(Real z);
    void setW(Real w);

	const Real& w() const;
	const Real& x() const;
	const Real& y() const;
	const Real& z() const;
    const Real& getW() const;
    const Real& getX() const;
    const Real& getY() const;
    const Real& getZ() const;

    Real& w();
	Real& x();
	Real& y();
	Real& z();
    Real& getW();
    Real& getX();
    Real& getY();
    Real& getZ();
    
    template <typename RealCastType>
    Quaternion<RealCastType> cast() const;

    Quaternion<Real>& operator = (const Quaternion<Real>& q);
	Quaternion<Real>& operator *= (const Quaternion<Real>& q);
    Quaternion<Real>& operator *= (Real scalar);
    Quaternion<Real> operator * (const Quaternion<Real>& q) const;
    Quaternion<Real> operator * (Real scalar) const;

    friend std::ostream& operator << <> (std::ostream& out, const Quaternion<Real>& q);

    static void FromAxisAngle(Real x, Real y, Real z, Real angle, Quaternion<Real>& q);
    static void FromAxisAngle(const Vector3<Real>& axis, Real angle, Quaternion<Real>& q);
    static void FromEulerAngles(Real yaw, Real pitch, Real roll, Quaternion<Real>& q);
    static void FromRotationMatrix(const RotationMatrix<Real>& rotationMatrix, Quaternion<Real>& q);
    static void Identity(Quaternion<Real>& q);
    static void Normalize(Quaternion<Real>& q);
    static void Conjugate(Quaternion<Real>& q);

    static Vector4<Real> ToVector(const Quaternion<Real>& q);
    static RotationMatrix<Real> ToRotationMatrix(const Quaternion<Real>& q);
    static Quaternion<Real> EulerRotationX(Real angle);
    static Quaternion<Real> EulerRotationY(Real angle);
    static Quaternion<Real> EulerRotationZ(Real angle);
    static Quaternion<Real> Multiply(const Quaternion<Real>& q, const Quaternion<Real>& p);
    static Quaternion<Real> Conjugate(const Quaternion<Real>& q);
    static Quaternion<Real> Normalize(const Quaternion<Real>& q);
    static Quaternion<Real> Identity();
    static Quaternion<Real> FromRotationMatrix(const RotationMatrix<Real>& rotationMatrix);
    static Quaternion<Real> FromEulerAngles(Real yaw, Real pitch, Real roll);
    static Quaternion<Real> FromAxisAngle(Real x, Real y, Real z, Real angle);
    static Quaternion<Real> FromAxisAngle(const Vector3<Real>& axis, Real angle);

    static Real Length(const Quaternion<Real>& q);
    static Real InnerProduct(const Quaternion<Real>& q, const Quaternion<Real>& p);

protected:
    Real data[COMPONENT_COUNT];
};

template <typename Real>
Quaternion<Real>::Quaternion(bool identity) {
    if (identity) this->identity();
    else std::memset(data, 0, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Quaternion<Real>::Quaternion(const Quaternion<Real>& q) {
    std::memcpy(data, q.data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Quaternion<Real>::Quaternion(const Vector3<Real>& axis, Real angle) {
    this->fromAxisAngle(axis, angle);
}

template <typename Real>
Quaternion<Real>::Quaternion(Real x, Real y, Real z, Real w) {
    this->set(x, y, z, w);
}

template <typename Real>
Quaternion<Real>::Quaternion(Real yaw, Real pitch, Real roll) {
    this->fromEulerAngles(yaw, pitch, roll);
}

template <typename Real>
Quaternion<Real>::Quaternion(const RotationMatrix<Real>& rotationMatrix) {
    this->fromRotationMatrix(rotationMatrix);
}

template <typename Real>
Quaternion<Real>::~Quaternion() {}

template <typename Real>
void Quaternion<Real>::normalize() {
    Normalize(*this);
}

template <typename Real>
void Quaternion<Real>::identity() {
    Identity(*this);
}

template <typename Real>
bool Quaternion<Real>::isIdentity() const {
    return (std::abs(data[X]) < 1e-6 &&
            std::abs(data[Y]) < 1e-6 &&
            std::abs(data[Z]) < 1e-6 &&
            std::abs(data[W] - Real(1)) < 1e-6);
}

template <typename Real>
void Quaternion<Real>::multiply(const Quaternion<Real>& q) {
    (*this) *= q;
}

template <typename Real>
void Quaternion<Real>::multiplyOnLeft(const Quaternion<Real>& q) {
    *this = q * (*this);
}

template <typename Real>
void Quaternion<Real>::multiplyOnRight(const Quaternion<Real>& q) {
    (*this) *= q;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::normalized() const {
    return Normalize(*this);
}

template <typename Real>
RotationMatrix<Real> Quaternion<Real>::toRotationMatrix() const {
    return ToRotationMatrix(*this);
}

template <typename Real>
Vector4<Real> Quaternion<Real>::toVector() const {
    return ToVector(*this);
}

template <typename Real>
Real Quaternion<Real>::length() const {
    return Length(*this);
}

template <typename Real>
void Quaternion<Real>::fromVector(const Vector4<Real>& v) {
    this->data[X] = v.x();
	this->data[Y] = v.y();
	this->data[Z] = v.z();
	this->data[W] = v.w();
}

template <typename Real>
void Quaternion<Real>::fromAxisAngle(const Vector3<Real>& axis, Real angle) {
    FromAxisAngle(axis, angle, *this);
}

template <typename Real>
void Quaternion<Real>::fromEulerAngles(Real yaw, Real pitch, Real roll) {
    FromEulerAngles(yaw, pitch, roll, *this);
}

template <typename Real>
void Quaternion<Real>::fromRotationMatrix(const RotationMatrix<Real>& rotationMatrix) {
    FromRotationMatrix(rotationMatrix, *this);
}

template <typename Real>
void Quaternion<Real>::fromEulerRotationX(Real angle) {
    this->fromEulerAngles(angle, Real(0), Real(0));
}

template <typename Real>
void Quaternion<Real>::fromEulerRotationY(Real angle) {
    this->fromEulerAngles(Real(0), angle, Real(0));
}

template <typename Real>
void Quaternion<Real>::fromEulerRotationZ(Real angle) {
    this->fromEulerAngles(Real(0), Real(0), angle);
}

template <typename Real>
void Quaternion<Real>::set(Real x, Real y, Real z, Real w) {
    this->data[X] = x;
	this->data[Y] = y;
	this->data[Z] = z;
	this->data[W] = w;
}

template <typename Real>
void Quaternion<Real>::set(const Vector4<Real>& v) {
    this->fromVector(v);
}

template <typename Real> void Quaternion<Real>::setX(Real x) { data[X] = x; }
template <typename Real> void Quaternion<Real>::setY(Real y) { data[Y] = y; }
template <typename Real> void Quaternion<Real>::setZ(Real z) { data[Z] = z; }
template <typename Real> void Quaternion<Real>::setW(Real w) { data[W] = w; }

template <typename Real> const Real& Quaternion<Real>::w() const { return data[W]; }
template <typename Real> const Real& Quaternion<Real>::x() const { return data[X]; }
template <typename Real> const Real& Quaternion<Real>::y() const { return data[Y]; }
template <typename Real> const Real& Quaternion<Real>::z() const { return data[Z]; }
template <typename Real> const Real& Quaternion<Real>::getW() const { return data[W]; }
template <typename Real> const Real& Quaternion<Real>::getX() const { return data[X]; }
template <typename Real> const Real& Quaternion<Real>::getY() const { return data[Y]; }
template <typename Real> const Real& Quaternion<Real>::getZ() const { return data[Z]; }

template <typename Real> Real& Quaternion<Real>::w() { return data[W]; }
template <typename Real> Real& Quaternion<Real>::x() { return data[X]; }
template <typename Real> Real& Quaternion<Real>::y() { return data[Y]; }
template <typename Real> Real& Quaternion<Real>::z() { return data[Z]; }
template <typename Real> Real& Quaternion<Real>::getW() { return data[W]; }
template <typename Real> Real& Quaternion<Real>::getX() { return data[X]; }
template <typename Real> Real& Quaternion<Real>::getY() { return data[Y]; }
template <typename Real> Real& Quaternion<Real>::getZ() { return data[Z]; }

template <typename Real>
template <typename RealCastType>
Quaternion<RealCastType> Quaternion<Real>::cast() const {
    return Quaternion<RealCastType>(
        static_cast<RealCastType>(data[X]),
        static_cast<RealCastType>(data[Y]),
        static_cast<RealCastType>(data[Z]),
        static_cast<RealCastType>(data[W])
    );
}

template <typename Real>
Quaternion<Real>& Quaternion<Real>::operator=(const Quaternion<Real>& q) {
    if (this != &q) {
        std::memcpy(data, q.data, COMPONENT_COUNT * sizeof(Real));
    }
    return *this;
}

template <typename Real>
Quaternion<Real>& Quaternion<Real>::operator*=(const Quaternion<Real>& q) {
    Real temp_w = data[W] * q.data[W] - data[X] * q.data[X] - data[Y] * q.data[Y] - data[Z] * q.data[Z];
    Real temp_x = data[W] * q.data[X] + data[X] * q.data[W] + data[Y] * q.data[Z] - data[Z] * q.data[Y];
    Real temp_y = data[W] * q.data[Y] - data[X] * q.data[Z] + data[Y] * q.data[W] + data[Z] * q.data[X];
    Real temp_z = data[W] * q.data[Z] + data[X] * q.data[Y] - data[Y] * q.data[X] + data[Z] * q.data[W];
    this->set(temp_x, temp_y, temp_z, temp_w);
    return *this;
}

template <typename Real>
Quaternion<Real>& Quaternion<Real>::operator*=(Real scalar) {
    data[X] *= scalar; data[Y] *= scalar; data[Z] *= scalar; data[W] *= scalar;
    return *this;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::operator*(const Quaternion<Real>& q) const {
    return Multiply(*this, q);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::operator*(Real scalar) const {
    Quaternion<Real> result = *this;
    result *= scalar;
    return result;
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const Quaternion<Real>& q) {
    out << "[x:" << q.x() << " y:" << q.y() << " z:" << q.z() << " w:" << q.w() << "]";
    return out;
}

template <typename Real>
void Quaternion<Real>::FromAxisAngle(Real x, Real y, Real z, Real angle, Quaternion<Real>& q) {
    Real halfAngle = angle * Real(0.5);
    Real sinHalfAngle = std::sin(halfAngle);
    q.set(x * sinHalfAngle, y * sinHalfAngle, z * sinHalfAngle, std::cos(halfAngle));
}

template <typename Real>
void Quaternion<Real>::FromAxisAngle(const Vector3<Real>& axis, Real angle, Quaternion<Real>& q) {
    FromAxisAngle(axis.x(), axis.y(), axis.z(), angle, q);
}

template <typename Real>
void Quaternion<Real>::FromEulerAngles(Real yaw_z, Real pitch_y, Real roll_x, Quaternion<Real>& q) {
    Real half_yaw = yaw_z * Real(0.5);
    Real half_pitch = pitch_y * Real(0.5);
    Real half_roll = roll_x * Real(0.5);
    Real cos_y = std::cos(half_yaw);
    Real sin_y = std::sin(half_yaw);
    Real cos_p = std::cos(half_pitch);
    Real sin_p = std::sin(half_pitch);
    Real cos_r = std::cos(half_roll);
    Real sin_r = std::sin(half_roll);

    q.set(cos_y * sin_p * cos_r + sin_y * cos_p * sin_r,
          cos_y * cos_p * sin_r - sin_y * sin_p * cos_r,
          sin_y * cos_p * cos_r + cos_y * sin_p * sin_r,
          cos_y * cos_p * cos_r - sin_y * sin_p * sin_r);
}

template <typename Real>
void Quaternion<Real>::FromRotationMatrix(const RotationMatrix<Real>& m, Quaternion<Real>& q) {
    Real trace = m(0, 0) + m(1, 1) + m(2, 2);
    if (trace > 0) {
        Real S = std::sqrt(trace + Real(1.0)) * 2;
        q.set((m(2, 1) - m(1, 2)) / S,
              (m(0, 2) - m(2, 0)) / S,
              (m(1, 0) - m(0, 1)) / S,
              Real(0.25) * S);
    } else if ((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2))) {
        Real S = std::sqrt(Real(1.0) + m(0, 0) - m(1, 1) - m(2, 2)) * 2;
        q.set(Real(0.25) * S,
              (m(0, 1) + m(1, 0)) / S,
              (m(0, 2) + m(2, 0)) / S,
              (m(2, 1) - m(1, 2)) / S);
    } else if (m(1, 1) > m(2, 2)) {
        Real S = std::sqrt(Real(1.0) + m(1, 1) - m(0, 0) - m(2, 2)) * 2;
        q.set((m(0, 1) + m(1, 0)) / S,
              Real(0.25) * S,
              (m(1, 2) + m(2, 1)) / S,
              (m(0, 2) - m(2, 0)) / S);
    } else {
        Real S = std::sqrt(Real(1.0) + m(2, 2) - m(0, 0) - m(1, 1)) * 2;
        q.set((m(0, 2) + m(2, 0)) / S,
              (m(1, 2) + m(2, 1)) / S,
              Real(0.25) * S,
              (m(1, 0) - m(0, 1)) / S);
    }
}

template <typename Real>
void Quaternion<Real>::Identity(Quaternion<Real>& q) {
    q.set(Real(0), Real(0), Real(0), Real(1));
}

template <typename Real>
void Quaternion<Real>::Normalize(Quaternion<Real>& q) {
    Real len = Length(q);
    if ( std::abs(len) > 1e-6 ) {
        q *= (Real(1) / len);
    }
}

template <typename Real>
void Quaternion<Real>::Conjugate(Quaternion<Real>& q) {
    q.set(-q.x(), -q.y(), -q.z(), q.w());
}

template <typename Real>
Vector4<Real> Quaternion<Real>::ToVector(const Quaternion<Real>& q) {
    return Vector4<Real>(q.x(), q.y(), q.z(), q.w());
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::EulerRotationX(Real angle) {
    return FromEulerAngles(angle, Real(0), Real(0));
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::EulerRotationY(Real angle) {
    return FromEulerAngles(Real(0), angle, Real(0));
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::EulerRotationZ(Real angle) {
    return FromEulerAngles(Real(0), Real(0), angle);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Multiply(const Quaternion<Real>& q, const Quaternion<Real>& p) {
    Quaternion<Real> result = q;
    result *= p;
    return result;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Conjugate(const Quaternion<Real>& q) {
    Quaternion<Real> result = q;
    Conjugate(result);
    return result;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Normalize(const Quaternion<Real>& q) {
    Quaternion<Real> result = q;
    Normalize(result);
    return result;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::Identity() {
    return Quaternion<Real>(true);
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::FromRotationMatrix(const RotationMatrix<Real>& m) {
    Quaternion<Real> q;
    FromRotationMatrix(m, q);
    return q;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::FromEulerAngles(Real yaw, Real pitch, Real roll) {
    Quaternion<Real> q;
    FromEulerAngles(yaw, pitch, roll, q);
    return q;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::FromAxisAngle(Real x, Real y, Real z, Real angle) {
    Quaternion<Real> q;
    FromAxisAngle(x, y, z, angle, q);
    return q;
}

template <typename Real>
Quaternion<Real> Quaternion<Real>::FromAxisAngle(const Vector3<Real>& axis, Real angle) {
    Quaternion<Real> q;
    FromAxisAngle(axis, angle, q);
    return q;
}

template <typename Real>
RotationMatrix<Real> Quaternion<Real>::ToRotationMatrix(const Quaternion<Real>& q) {
    RotationMatrix<Real> rot;
    Real x = q.x(), y = q.y(), z = q.z(), w = q.w();
    Real xx = x * x, yy = y * y, zz = z * z;
    Real xy = x * y, xz = x * z, yz = y * z;
    Real wx = w * x, wy = w * y, wz = w * z;
    
    rot(0, 0) = 1 - 2 * (yy + zz);
    rot(0, 1) = 2 * (xy - wz);
    rot(0, 2) = 2 * (xz + wy);
    rot(1, 0) = 2 * (xy + wz);
    rot(1, 1) = 1 - 2 * (xx + zz);
    rot(1, 2) = 2 * (yz - wx);
    rot(2, 0) = 2 * (xz - wy);
    rot(2, 1) = 2 * (yz + wx);
    rot(2, 2) = 1 - 2 * (xx + yy);
    
    return rot;
}

template <typename Real>
Real Quaternion<Real>::Length(const Quaternion<Real>& q) {
    return std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
}

template <typename Real>
Real Quaternion<Real>::InnerProduct(const Quaternion<Real>& q, const Quaternion<Real>& p) {
    return q.x() * p.x() + q.y() * p.y() + q.z() * p.z() + q.w() * p.w();
}

typedef Quaternion<float> Quaternionf;
typedef Quaternion<double> Quaterniond;
typedef Quaternion<long double> Quaternionld;

}

#endif
