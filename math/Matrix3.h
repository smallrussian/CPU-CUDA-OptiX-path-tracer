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
#ifndef MATRIX_3_H
#define MATRIX_3_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <iomanip>
#include <stdexcept>

#include "Vector3.h"
#include "Vector4.h"

namespace graphics {

template <typename Real>
class Matrix3;

template <typename Real>
Vector3<Real> operator * (const Matrix3<Real>& m, const Vector3<Real>& v);

template <typename Real>
std::ostream& operator << (std::ostream& out, const Matrix3<Real>& m);

/* Matrix3: Representation of any numerical 3x3 matrix.
 * [a11 a12 a13]
 * [a21 a22 a23]
 * [a31 a32 a33]
 */
template <typename Real>
class Matrix3 {
    static_assert(std::is_integral<Real>::value || std::is_floating_point<Real>::value, 
    "[Matrix3:Type] Error: Matrix3 Real type must be an integral or floating point numerical representation.");

    enum Element { A_11, A_21, A_31, 
                   A_12, A_22, A_32,
                   A_13, A_23, A_33,
                   COMPONENT_COUNT };
public:
    Matrix3(bool identity = true);
    Matrix3(const Real* const data);
    Matrix3(const Matrix3<Real>& m);
    Matrix3(Real a11, Real a12, Real a13, 
            Real a21, Real a22, Real a23, 
            Real a31, Real a32, Real a33, bool colMajor = true);
    ~Matrix3();

    void set(const Matrix3<Real>& m);
    void set(const Real* const data);
    void set(std::size_t row, std::size_t col, Real value);
    void set(Real a11, Real a12, Real a13, 
             Real a21, Real a22, Real a23, 
             Real a31, Real a32, Real a33, bool colMajor = true);

    void setRow(std::size_t i, Real x, Real y, Real z);
    void setRow(std::size_t i, const Vector3<Real>& row);
    void setColumn(std::size_t i, Real x, Real y, Real z);
    void setColumn(std::size_t i, const Vector3<Real>& column);

    bool isZero(Real epsilon) const noexcept;
    bool isIdentity(Real epsilon) const noexcept;
    bool isEquivalent(const Matrix3<Real>& m, Real epsilon) const noexcept;

    void zero() noexcept;
    void transpose() noexcept;
    void identity() noexcept;

    void clear(bool makeIdentity = true) noexcept;
    void toRawMatrix(Real* const matrix, bool colMajor = true) const;
    void getData(Real* const matrix, bool colMajor = true) const;
    
    Real determinant() const;
    Matrix3<Real> inverse() const;
    Matrix3<Real> inversed() const;
    Matrix3<Real> transposed() const;

    template <typename RealCastType>
    Matrix3<RealCastType> cast() const;

    Real& get(std::size_t row, std::size_t col);
    const Real& get(std::size_t row, std::size_t col) const;
    Vector3<Real> getRow(std::size_t i) const;
    Vector3<Real> getColumn(std::size_t i) const;

    Matrix3<Real> toTranspose() const;
    Matrix3<Real> toInverse() const;
    Matrix3<Real> apply(const Vector3<Real>& v, bool colMajor = true) const;
    Vector3<Real> applyTo(const Vector3<Real>& v) const;
	Vector4<Real> applyTo(const Vector4<Real>& v) const;

    const Real* constData() const;
    operator const Real* () const;

    Real& operator () (std::size_t row, std::size_t col);
    const Real& operator () (std::size_t row, std::size_t col) const;

    Real& operator [] (std::size_t index);
    const Real& operator [] (std::size_t index) const;

	friend Vector3<Real> operator * <> (const Matrix3<Real>& m, const Vector3<Real>& v);
    friend std::ostream& operator << <> (std::ostream& out, const Matrix3<Real>& m);

    bool operator == (const Matrix3<Real>& m) const;
	bool operator != (const Matrix3<Real>& m) const;

    Matrix3<Real>& operator = (const Matrix3<Real>& m);
	Matrix3<Real>& operator = (const Real* data);

    Vector3<Real> operator * (const Vector3<Real>& v) const;
	Matrix3<Real> operator * (const Matrix3<Real>& m) const;
    Matrix3<Real>& operator *= (const Matrix3<Real>& m);

    static void ToRawMatrix(const Matrix3<Real>& m, Real* const matrix, bool colMajor = true);
    static void Clear(Matrix3<Real>& m);
    static void Identity(Matrix3<Real>& m);
    static void Zero(Matrix3<Real>& m);
    static Real Determinant(const Matrix3<Real>& m);
    static Matrix3<Real> Multiply(const Matrix3<Real>& a, const Matrix3<Real>& b);
    static Matrix3<Real> Transpose(const Matrix3<Real>& m);
    static Matrix3<Real> Inverse(const Matrix3<Real>& m);

    static Matrix3<Real> Zero();
    static Matrix3<Real> Identity();

protected:
    static constexpr int ROW_COUNT = 3;
    static constexpr int COL_COUNT = 3;

    Real data[COMPONENT_COUNT];
};

template <typename Real>
Matrix3<Real>::Matrix3(bool identity) {
    if (identity) this->identity();
    else this->zero();
}

template <typename Real>
Matrix3<Real>::Matrix3(const Real* const data) {
    std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Matrix3<Real>::Matrix3(const Matrix3<Real>& m) {
    std::memcpy(this->data, m.data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Matrix3<Real>::Matrix3(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23, Real a31, Real a32, Real a33, bool colMajor) {
    this->set(a11, a12, a13, a21, a22, a23, a31, a32, a33, colMajor);
}

template <typename Real>
Matrix3<Real>::~Matrix3() {}

template <typename Real>
Real& Matrix3<Real>::operator () (std::size_t row, std::size_t col) {
    if (row >= ROW_COUNT || col >= COL_COUNT) throw std::out_of_range("[Matrix3::()] Index out of bounds.");
    return data[col * ROW_COUNT + row];
}

template <typename Real>
const Real& Matrix3<Real>::operator () (std::size_t row, std::size_t col) const {
    if (row >= ROW_COUNT || col >= COL_COUNT) throw std::out_of_range("[Matrix3::()] Index out of bounds.");
    return data[col * ROW_COUNT + row];
}

template <typename Real>
Real& Matrix3<Real>::get(std::size_t row, std::size_t col) {
    return (*this)(row, col);
}

template <typename Real>
const Real& Matrix3<Real>::get(std::size_t row, std::size_t col) const {
    return (*this)(row, col);
}

template <typename Real>
Real& Matrix3<Real>::operator [] (std::size_t index) {
    if (index >= COMPONENT_COUNT) throw std::out_of_range("[Matrix3::[]] Index out of bounds.");
    return data[index];
}

template <typename Real>
const Real& Matrix3<Real>::operator [] (std::size_t index) const {
    if (index >= COMPONENT_COUNT) throw std::out_of_range("[Matrix3::[]] Index out of bounds.");
    return data[index];
}

template <typename Real>
void Matrix3<Real>::set(const Matrix3<Real>& m) {
    std::memcpy(this->data, m.data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
void Matrix3<Real>::set(const Real* const data) {
    std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
void Matrix3<Real>::set(std::size_t row, std::size_t col, Real value) {
    (*this)(row, col) = value;
}

template <typename Real>
void Matrix3<Real>::set(Real a11, Real a12, Real a13, Real a21, Real a22, Real a23, Real a31, Real a32, Real a33, bool colMajor) {
    if (colMajor) {
        data[A_11] = a11; data[A_12] = a12; data[A_13] = a13;
        data[A_21] = a21; data[A_22] = a22; data[A_23] = a23;
        data[A_31] = a31; data[A_32] = a32; data[A_33] = a33;
    } else {
        data[A_11] = a11; data[A_21] = a12; data[A_31] = a13;
        data[A_12] = a21; data[A_22] = a22; data[A_32] = a23;
        data[A_13] = a31; data[A_23] = a32; data[A_33] = a33;
    }
}

template <typename Real>
void Matrix3<Real>::setRow(std::size_t i, Real x, Real y, Real z) {
    if (i >= ROW_COUNT) throw std::out_of_range("[Matrix3::setRow] Index out of bounds.");
    (*this)(i, 0) = x;
    (*this)(i, 1) = y;
    (*this)(i, 2) = z;
}

template <typename Real>
void Matrix3<Real>::setRow(std::size_t i, const Vector3<Real>& row) {
    setRow(i, row.x(), row.y(), row.z());
}

template <typename Real>
void Matrix3<Real>::setColumn(std::size_t i, Real x, Real y, Real z) {
    if (i >= COL_COUNT) throw std::out_of_range("[Matrix3::setColumn] Index out of bounds.");
    (*this)(0, i) = x;
    (*this)(1, i) = y;
    (*this)(2, i) = z;
}

template <typename Real>
void Matrix3<Real>::setColumn(std::size_t i, const Vector3<Real>& column) {
    setColumn(i, column.x(), column.y(), column.z());
}

template <typename Real>
bool Matrix3<Real>::isZero(Real epsilon) const noexcept {
    for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
        if (std::abs(data[i]) > epsilon) return false;
    }
    return true;
}

template <typename Real>
bool Matrix3<Real>::isIdentity(Real epsilon) const noexcept {
    for ( int i = 0; i < ROW_COUNT; ++i ) {
        for ( int j = 0; j < COL_COUNT; ++j ) {
            Real expected = (i == j) ? Real(1) : Real(0);
            if ( std::abs((*this)(i, j) - expected) > epsilon ) return false;
        }
    }
    return true;
}

template <typename Real>
bool Matrix3<Real>::isEquivalent(const Matrix3<Real>& m, Real epsilon) const noexcept {
    for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
        if (std::abs(data[i] - m.data[i]) > epsilon) return false;
    }
    return true;
}

template <typename Real>
void Matrix3<Real>::zero() noexcept {
    std::memset(data, 0, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
void Matrix3<Real>::transpose() noexcept {
    std::swap((*this)(0, 1), (*this)(1, 0));
    std::swap((*this)(0, 2), (*this)(2, 0));
    std::swap((*this)(1, 2), (*this)(2, 1));
}

template <typename Real>
void Matrix3<Real>::identity() noexcept {
    zero();
    (*this)(0, 0) = Real(1);
    (*this)(1, 1) = Real(1);
    (*this)(2, 2) = Real(1);
}

template <typename Real>
void Matrix3<Real>::clear(bool makeIdentity) noexcept {
    if ( makeIdentity ) identity();
    else zero();
}

template <typename Real>
void Matrix3<Real>::toRawMatrix(Real* const matrix, bool colMajor) const {
    if (matrix == nullptr) return;
    if (colMajor) {
        std::memcpy(matrix, data, COMPONENT_COUNT * sizeof(Real));
    } else {
        Matrix3<Real> temp = transposed();
        std::memcpy(matrix, temp.data, COMPONENT_COUNT * sizeof(Real));
    }
}

template <typename Real>
void Matrix3<Real>::getData(Real* const matrix, bool colMajor) const {
    this->toRawMatrix(matrix, colMajor);
}

template <typename Real>
Real Matrix3<Real>::determinant() const {
    return Determinant(*this);
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::inverse() const {
    return Inverse(*this);
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::inversed() const {
    return inverse();
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::transposed() const {
    return Transpose(*this);
}

template <typename Real>
template <typename RealCastType>
Matrix3<RealCastType> Matrix3<Real>::cast() const {
    Matrix3<RealCastType> result;
    for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
        result.data[i] = static_cast<RealCastType>(data[i]);
    }
    return result;
}

template <typename Real>
Vector3<Real> Matrix3<Real>::getRow(std::size_t i) const {
    if ( i >= ROW_COUNT ) throw std::out_of_range("[Matrix3::getRow] Index out of bounds.");
    return Vector3<Real>((*this)(i, 0), (*this)(i, 1), (*this)(i, 2));
}

template <typename Real>
Vector3<Real> Matrix3<Real>::getColumn(std::size_t i) const {
    if ( i >= COL_COUNT ) throw std::out_of_range("[Matrix3::getColumn] Index out of bounds.");
    return Vector3<Real>((*this)(0, i), (*this)(1, i), (*this)(2, i));
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::toTranspose() const {
    return transposed();
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::toInverse() const {
    return inverse();
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::apply(const Vector3<Real>& v, bool colMajor) const {
    Matrix3<Real> m = transposed();
    m.setRow(0, m(0, 0) * v.x(), m(0, 1) * v.x(), m(0, 2) * v.x());
    m.setRow(1, m(1, 0) * v.y(), m(1, 1) * v.y(), m(1, 2) * v.y());
    m.setRow(2, m(2, 0) * v.z(), m(2, 1) * v.z(), m(2, 2) * v.z());
    if (colMajor) m.transpose();
    return m;
}

template <typename Real>
Vector3<Real> Matrix3<Real>::applyTo(const Vector3<Real>& v) const {
    return Vector3<Real>(
        (*this)(0, 0) * v.x() + (*this)(0, 1) * v.y() + (*this)(0, 2) * v.z(),
        (*this)(1, 0) * v.x() + (*this)(1, 1) * v.y() + (*this)(1, 2) * v.z(),
        (*this)(2, 0) * v.x() + (*this)(2, 1) * v.y() + (*this)(2, 2) * v.z()
    );
}

template <typename Real>
Vector4<Real> Matrix3<Real>::applyTo(const Vector4<Real>& v) const {
    return Vector4<Real>(
        (*this)(0, 0) * v.x() + (*this)(0, 1) * v.y() + (*this)(0, 2) * v.z(),
        (*this)(1, 0) * v.x() + (*this)(1, 1) * v.y() + (*this)(1, 2) * v.z(),
        (*this)(2, 0) * v.x() + (*this)(2, 1) * v.y() + (*this)(2, 2) * v.z(),
        v.w()
    );
}

template <typename Real>
const Real* Matrix3<Real>::constData() const {
    return data;
}

template <typename Real>
Matrix3<Real>::operator const Real* () const {
    return data;
}

template <typename Real>
Vector3<Real> operator * (const Matrix3<Real>& m, const Vector3<Real>& v) {
    return m.applyTo(v);
}

template <typename Real>
std::ostream& operator << (std::ostream& out, const Matrix3<Real>& m) {
    out << std::fixed << std::setprecision(6);
    out << "[" << m(0, 0) << "\t" << m(0, 1) << "\t" << m(0, 2) << "]\n";
    out << "[" << m(1, 0) << "\t" << m(1, 1) << "\t" << m(1, 2) << "]\n";
    out << "[" << m(2, 0) << "\t" << m(2, 1) << "\t" << m(2, 2) << "]\n";
    return out;
}

template <typename Real>
bool Matrix3<Real>::operator == (const Matrix3<Real>& m) const {
    return this->isEquivalent(m, static_cast<Real>(1e-6));
}

template <typename Real>
bool Matrix3<Real>::operator != (const Matrix3<Real>& m) const {
    return !(*this == m);
}

template <typename Real>
Matrix3<Real>& Matrix3<Real>::operator = (const Matrix3<Real>& m) {
    if ( this != &m ) {
        std::memcpy(data, m.data, COMPONENT_COUNT * sizeof(Real));
    }
    return *this;
}

template <typename Real>
Matrix3<Real>& Matrix3<Real>::operator = (const Real* data) {
    if ( this->data != data ) {
        std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
    }
    return *this;
}

template <typename Real>
Vector3<Real> Matrix3<Real>::operator * (const Vector3<Real>& v) const {
    return applyTo(v);
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::operator * (const Matrix3<Real>& m) const {
    return Multiply(*this, m);
}

template <typename Real>
Matrix3<Real>& Matrix3<Real>::operator *= (const Matrix3<Real>& m) {
    *this = Multiply(*this, m);
    return *this;
}

template <typename Real>
void Matrix3<Real>::ToRawMatrix(const Matrix3<Real>& m, Real* const matrix, bool colMajor) {
    m.toRawMatrix(matrix, colMajor);
}

template <typename Real>
void Matrix3<Real>::Clear(Matrix3<Real>& m) {
    m.clear(false);
}

template <typename Real>
void Matrix3<Real>::Identity(Matrix3<Real>& m) {
    m.identity();
}

template <typename Real>
void Matrix3<Real>::Zero(Matrix3<Real>& m) {
    m.zero();
}

template <typename Real>
Real Matrix3<Real>::Determinant(const Matrix3<Real>& m) {
    return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) -
           m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
           m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::Multiply(const Matrix3<Real>& a, const Matrix3<Real>& b) {
    Matrix3<Real> result(false);
    for ( int i = 0; i < ROW_COUNT; ++i ) {
        for  (int j = 0; j < COL_COUNT; ++j ) {
            for ( int k = 0; k < COL_COUNT; ++k ) {
                result(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return result;
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::Transpose(const Matrix3<Real>& m) {
    Matrix3<Real> result = m;
    result.transpose();
    return result;
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::Inverse(const Matrix3<Real>& m) {
    Real d = Determinant(m);
    if ( std::abs(d) < 1e-6 ) {
		std::cerr << "[Matrix3:Inverse] Notice: Singular matrix." << std::endl;
        return Matrix3<Real>(true);
    }
    
    Real inv_det = Real(1) / d;
    Matrix3<Real> result;
    
    // Adjugate transpose.
    result(0, 0) = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * inv_det;
    result(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
    result(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
    result(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * inv_det;
    result(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
    result(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * inv_det;
    result(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * inv_det;
    result(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * inv_det;
    result(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * inv_det;

    return result;
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::Zero() {
    return Matrix3<Real>(false);
}

template <typename Real>
Matrix3<Real> Matrix3<Real>::Identity() {
    return Matrix3<Real>(true);
}

typedef Matrix3<float> Matrix3f;
typedef Matrix3<double> Matrix3d;
typedef Matrix3<long> Matrix3l;
typedef Matrix3<int> Matrix3i;
typedef Matrix3<short> Matrix3s;
typedef Matrix3<float> Mat3f;
typedef Matrix3<double> Mat3d;
typedef Matrix3<long> Mat3l;
typedef Matrix3<int> Mat3i;
typedef Matrix3<short> Mat3s;
typedef Matrix3<float> Mat3;

}

#endif
