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
#ifndef MATRIX_4_H
#define MATRIX_4_H

#include <iostream>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <cstring>

#include "Mathematics.h"
#include "Matrix3.h"
#include "Vector3.h"
#include "Vector4.h"

namespace graphics {

template <typename Real>
class Matrix4;

template <typename Real>
Vector3<Real> operator * (const Matrix4<Real>& m, const Vector3<Real>& v);

template <typename Real>
Vector4<Real> operator * (const Matrix4<Real>& m, const Vector4<Real>& v);

template <typename Real>
std::ostream& operator << (std::ostream& out, const Matrix4<Real>& m);

/*
 * Matrix4: Representation of any numerical 4x4 matrix.
 * Internal memory layout is Column-Major, consistent with OpenGL.
 * m[0] m[4] m[8]  m[12]
 * m[1] m[5] m[9]  m[13]
 * m[2] m[6] m[10] m[14]
 * m[3] m[7] m[11] m[15]
 */
template <typename Real>
class Matrix4 {
	static_assert(std::is_integral<Real>::value || std::is_floating_point<Real>::value,
		"[Matrix4:Type] Error: Matrix4 Real type must be an integral or floating point numerical representation.");

	// Enum defines memory layout for a column-major matrix
	enum Element {
		A_11, A_21, A_31, A_41, // Column 1
		A_12, A_22, A_32, A_42, // Column 2
		A_13, A_23, A_33, A_43, // Column 3
		A_14, A_24, A_34, A_44, // Column 4
		COMPONENT_COUNT
	};

public:
	Matrix4(bool identity = true);
	Matrix4(const Real* const data);
	Matrix4(const Matrix3<Real>& m, bool homogeneous = true);
	Matrix4(const Matrix4<Real>& m);
	Matrix4(Real a11, Real a12, Real a13, Real a14,
		    Real a21, Real a22, Real a23, Real a24,
		    Real a31, Real a32, Real a33, Real a34,
		    Real a41, Real a42, Real a43, Real a44, bool colMajor = true);
	~Matrix4();

	void set(const Matrix3<Real>& m, bool homogeneous = true);
	void set(const Matrix4<Real>& m);
	void set(const Real* const data);
	void set(std::size_t row, std::size_t col, Real value);
	void set(Real a11, Real a12, Real a13, Real a14,
		    Real a21, Real a22, Real a23, Real a24,
		    Real a31, Real a32, Real a33, Real a34,
		    Real a41, Real a42, Real a43, Real a44, bool colMajor = true);

	void setRow(std::size_t i, Real c1, Real c2, Real c3, Real c4);
	void setRow(std::size_t i, const Vector4<Real>& row);
	void setColumn(std::size_t i, Real r1, Real r2, Real r3, Real r4);
	void setColumn(std::size_t i, const Vector4<Real>& column);
	void setIdentity();

	bool isZero(Real epsilon) const;
	bool isIdentity(Real epsilon) const;
	bool isEquivalent(const Matrix4<Real>& m, Real epsilon) const;

	void zero();
	void transpose();
	void identity();
	void invert();
	void clear(bool makeIdentity = true);

	void toRawMatrix(Real* const matrix, bool colMajor = true) const;
	void getData(Real* const matrix, bool colMajor = true) const;
	Real determinant() const;
	Matrix4<Real> inverse() const;
    Matrix4<Real> inversed() const;
	Matrix4<Real> transposed() const;
    Matrix4<Real> toTranspose() const;
    Matrix4<Real> toInverse() const;

	template <typename RealCastType>
	Matrix4<RealCastType> cast() const;

	Real& get(std::size_t row, std::size_t col);
	const Real& get(std::size_t row, std::size_t col) const;
	Vector4<Real> getRow(std::size_t i) const;
	Vector4<Real> getColumn(std::size_t i) const;
    const Real* constData() const;

	Matrix4<Real> apply(const Vector3<Real>& v, bool colMajor = true) const;
	Matrix4<Real> apply(const Vector4<Real>& v, bool colMajor = true) const;
	Vector3<Real> applyTo(const Vector3<Real>& v) const;
	Vector4<Real> applyTo(const Vector4<Real>& v) const;

    operator const Real* () const;
	Real& operator () (std::size_t row, std::size_t col);
	const Real& operator () (std::size_t row, std::size_t col) const;
	Real& operator [] (std::size_t index);
	const Real& operator [] (std::size_t index) const;

	friend Vector3<Real> operator * <> (const Matrix4<Real>& m, const Vector3<Real>& v);
	friend Vector4<Real> operator * <> (const Matrix4<Real>& m, const Vector4<Real>& v);
	friend std::ostream& operator << <> (std::ostream& out, const Matrix4<Real>& m);

	bool operator == (const Matrix4<Real>& m) const;
	bool operator != (const Matrix4<Real>& m) const;

	Matrix4<Real>& operator = (const Matrix4<Real>& m);
	Matrix4<Real>& operator = (const Real* data);

	Vector3<Real> operator * (const Vector3<Real>& v) const;
	Vector4<Real> operator * (const Vector4<Real>& v) const;
	Matrix4<Real> operator * (const Matrix4<Real>& m) const;
	Matrix4<Real>& operator *= (const Matrix4<Real>& m);

	static void ToRawMatrix(const Matrix4<Real>& m, Real* const matrix, bool colMajor = true);
	static void Clear(Matrix4<Real>& m);
	static void Identity(Matrix4<Real>& m);
	static void Zero(Matrix4<Real>& m);
	static Real Determinant(const Matrix4<Real>& m);
	static Matrix4<Real> Multiply(const Matrix4<Real>& a, const Matrix4<Real>& b);
	static Matrix4<Real> Transpose(const Matrix4<Real>& m);
	static Matrix4<Real> Inverse(const Matrix4<Real>& m);
	static Matrix4<Real> LookAt(const Vector3<Real>& eye, const Vector3<Real>& lookat, const Vector3<Real>& up);
	static Matrix4<Real> LookAt(Real eyex, Real eyey, Real eyez, Real atx, Real aty, Real atz, Real upx, Real upy, Real upz);
	static Matrix4<Real> Perspective(Real fovY, Real aspect, Real nearPlane, Real farPlane);
	static Matrix4<Real> Zero();
	static Matrix4<Real> Identity();
	static Matrix3<Real> ToMatrix3(const Matrix4<Real>& matrix);
	static Matrix4<Real> Translation(const Vector3<Real>& t);
	static Matrix4<Real> Scale(const Vector3<Real>& s);
	static Matrix4<Real> RotationX(Real angleRadians);
	static Matrix4<Real> RotationY(Real angleRadians);
	static Matrix4<Real> RotationZ(Real angleRadians);

protected:
	static constexpr int ROW_COUNT = 4;
	static constexpr int COL_COUNT = 4;

	Real data[COMPONENT_COUNT];

private:
	static Real det(const Matrix4<Real>& matrix, Real* const adjoint);
};

template <typename Real>
Matrix4<Real>::Matrix4(bool identity) {
	if (identity) this->identity();
	else this->zero();
}

template <typename Real>
Matrix4<Real>::Matrix4(const Real* const data) {
	std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Matrix4<Real>::Matrix4(const Matrix3<Real>& m, bool homogeneous) {
	set(m, homogeneous);
}

template <typename Real>
Matrix4<Real>::Matrix4(const Matrix4<Real>& m) {
	std::memcpy(this->data, m.data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
Matrix4<Real>::Matrix4(Real a11, Real a12, Real a13, Real a14,
	Real a21, Real a22, Real a23, Real a24,
	Real a31, Real a32, Real a33, Real a34,
	Real a41, Real a42, Real a43, Real a44, bool colMajor) {
	set(a11, a12, a13, a14, a21, a22, a23, a24,
		a31, a32, a33, a34, a41, a42, a43, a44, colMajor);
}

template <typename Real>
Matrix4<Real>::~Matrix4() {}

template <typename Real>
Real& Matrix4<Real>::operator()(std::size_t row, std::size_t col) {
	if (row >= ROW_COUNT || col >= COL_COUNT) throw std::out_of_range("[Matrix4::()] Index out of bounds.");
	return data[col * ROW_COUNT + row];
}

template <typename Real>
const Real& Matrix4<Real>::operator()(std::size_t row, std::size_t col) const {
	if (row >= ROW_COUNT || col >= COL_COUNT) throw std::out_of_range("[Matrix4::()] Index out of bounds.");
	return data[col * ROW_COUNT + row];
}

template <typename Real>
Real& Matrix4<Real>::get(std::size_t row, std::size_t col) {
	return (*this)(row, col);
}

template <typename Real>
const Real& Matrix4<Real>::get(std::size_t row, std::size_t col) const {
	return (*this)(row, col);
}

template <typename Real>
Real& Matrix4<Real>::operator[](std::size_t index) {
	if (index >= COMPONENT_COUNT) throw std::out_of_range("[Matrix4::[]] Index out of bounds.");
	return data[index];
}

template <typename Real>
const Real& Matrix4<Real>::operator[](std::size_t index) const {
	if (index >= COMPONENT_COUNT) throw std::out_of_range("[Matrix4::[]] Index out of bounds.");
	return data[index];
}

template <typename Real>
void Matrix4<Real>::set(std::size_t row, std::size_t col, Real value) {
	(*this)(row, col) = value;
}

template <typename Real>
void Matrix4<Real>::set(Real a11, Real a12, Real a13, Real a14,
	Real a21, Real a22, Real a23, Real a24,
	Real a31, Real a32, Real a33, Real a34,
	Real a41, Real a42, Real a43, Real a44, bool colMajor) {

	if ( colMajor ) {
		// Input is column-major, so do a direct mapping
		data[A_11]=a11; data[A_21]=a21; data[A_31]=a31; data[A_41]=a41;
		data[A_12]=a12; data[A_22]=a22; data[A_32]=a32; data[A_42]=a42;
		data[A_13]=a13; data[A_23]=a23; data[A_33]=a33; data[A_43]=a43;
		data[A_14]=a14; data[A_24]=a24; data[A_34]=a34; data[A_44]=a44;
	}
	else {
		// Input is row-major, so transpose it on assignment
		data[A_11]=a11; data[A_21]=a12; data[A_31]=a13; data[A_41]=a14;
		data[A_12]=a21; data[A_22]=a22; data[A_32]=a23; data[A_42]=a24;
		data[A_13]=a31; data[A_23]=a32; data[A_33]=a33; data[A_43]=a34;
		data[A_14]=a41; data[A_24]=a42; data[A_34]=a43; data[A_44]=a44;
	}
}

template <typename Real>
void Matrix4<Real>::setRow(std::size_t i, Real c1, Real c2, Real c3, Real c4) {
	if ( i >= ROW_COUNT ) throw std::out_of_range("[Matrix4::setRow] Row index out of bounds.");
	(*this)(i, 0) = c1;
	(*this)(i, 1) = c2;
	(*this)(i, 2) = c3;
	(*this)(i, 3) = c4;
}

template <typename Real>
void Matrix4<Real>::setRow(std::size_t i, const Vector4<Real>& row) {
    this->setRow(i, row.w(), row.x(), row.y(), row.z());
}

template <typename Real>
void Matrix4<Real>::setColumn(std::size_t i, Real r1, Real r2, Real r3, Real r4) {
	if (i >= COL_COUNT) throw std::out_of_range("[Matrix4::setColumn] Column index out of bounds.");
	(*this)(0, i) = r1;
	(*this)(1, i) = r2;
	(*this)(2, i) = r3;
	(*this)(3, i) = r4;
}

template <typename Real>
void Matrix4<Real>::setColumn(std::size_t i, const Vector4<Real>& column) {
    this->setColumn(i, column.w(), column.x(), column.y(), column.z());
}

template <typename Real>
void Matrix4<Real>::setIdentity() {
	this->identity();
}

template <typename Real>
void Matrix4<Real>::set(const Matrix3<Real>& m, bool homogeneous) {
	zero();
	if ( homogeneous ) (*this)(3, 3) = Real(1);

	(*this)(0, 0) = m(0, 0); (*this)(0, 1) = m(0, 1); (*this)(0, 2) = m(0, 2);
	(*this)(1, 0) = m(1, 0); (*this)(1, 1) = m(1, 1); (*this)(1, 2) = m(1, 2);
	(*this)(2, 0) = m(2, 0); (*this)(2, 1) = m(2, 1); (*this)(2, 2) = m(2, 2);
}

template <typename Real>
void Matrix4<Real>::set(const Matrix4<Real>& m) {
	std::memcpy(data, m.data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
void Matrix4<Real>::set(const Real* const data) {
	std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
bool Matrix4<Real>::isZero(Real epsilon) const {
	for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
		if ( std::abs(data[i]) > epsilon ) return false;
	}
	return true;
}

template <typename Real>
bool Matrix4<Real>::isIdentity(Real epsilon) const {
	for ( int i = 0; i < ROW_COUNT; ++i ) {
		for ( int j = 0; j < COL_COUNT; ++j ) {
			Real expected = (i == j) ? Real(1) : Real(0);
			if ( std::abs((*this)(i, j) - expected) > epsilon ) return false;
		}
	}
	return true;
}

template <typename Real>
bool Matrix4<Real>::isEquivalent(const Matrix4<Real>& m, Real epsilon) const {
	for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
		if ( std::abs(data[i] - m.data[i]) > epsilon ) return false;
	}
	return true;
}

template <typename Real>
void Matrix4<Real>::zero() {
	std::memset(data, 0, COMPONENT_COUNT * sizeof(Real));
}

template <typename Real>
void Matrix4<Real>::transpose() {
	std::swap((*this)(0, 1), (*this)(1, 0));
	std::swap((*this)(0, 2), (*this)(2, 0));
	std::swap((*this)(0, 3), (*this)(3, 0));
	std::swap((*this)(1, 2), (*this)(2, 1));
	std::swap((*this)(1, 3), (*this)(3, 1));
	std::swap((*this)(2, 3), (*this)(3, 2));
}

template <typename Real>
void Matrix4<Real>::identity() {
	this->zero();
	(*this)(0, 0) = Real(1);
	(*this)(1, 1) = Real(1);
	(*this)(2, 2) = Real(1);
	(*this)(3, 3) = Real(1);
}

template <typename Real>
void Matrix4<Real>::invert() {
	*this = inverse();
}

template <typename Real>
void Matrix4<Real>::clear(bool makeIdentity) {
	if (makeIdentity) identity();
	else zero();
}

template <typename Real>
void Matrix4<Real>::toRawMatrix(Real* const matrix, bool colMajor) const {
	if ( matrix == nullptr ) return;
	if ( colMajor ) {
		std::memcpy(matrix, data, COMPONENT_COUNT * sizeof(Real));
	} else {
		Matrix4<Real> temp = transposed();
		std::memcpy(matrix, temp.data, COMPONENT_COUNT * sizeof(Real));
	}
}

template <typename Real>
void Matrix4<Real>::getData(Real* const matrix, bool colMajor) const {
	this->toRawMatrix(matrix, colMajor);
}

template <typename Real>
Real Matrix4<Real>::determinant() const {
	return Determinant(*this);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::inverse() const {
	return Inverse(*this);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::inversed() const {
    return inverse();
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::transposed() const {
	return Transpose(*this);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::toTranspose() const {
    return transposed();
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::toInverse() const {
    return inverse();
}

template <typename Real>
template <typename RealCastType>
Matrix4<RealCastType> Matrix4<Real>::cast() const {
	Matrix4<RealCastType> result;
	for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
		result.data[i] = static_cast<RealCastType>(data[i]);
	}
	return result;
}

template <typename Real>
Vector4<Real> Matrix4<Real>::getRow(std::size_t i) const {
	if ( i >= ROW_COUNT ) throw std::out_of_range("[Matrix4::getRow] Row index out of bounds.");
	return Vector4<Real>((*this)(i, 0), (*this)(i, 1), (*this)(i, 2), (*this)(i, 3));
}

template <typename Real>
Vector4<Real> Matrix4<Real>::getColumn(std::size_t i) const {
	if ( i >= COL_COUNT ) throw std::out_of_range("[Matrix4::getColumn] Column index out of bounds.");
	return Vector4<Real>((*this)(0, i), (*this)(1, i), (*this)(2, i), (*this)(3, i));
}

template <typename Real>
const Real* Matrix4<Real>::constData() const {
	return data;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::apply(const Vector3<Real>& v, bool colMajor) const {
	Matrix4<Real> m = transposed();
	m.setRow(0, m(0, 0) * v.x(), m(0, 1) * v.x(), m(0, 2) * v.x(), m(0, 3) * v.x());
	m.setRow(1, m(1, 0) * v.y(), m(1, 1) * v.y(), m(1, 2) * v.y(), m(1, 3) * v.y());
	m.setRow(2, m(2, 0) * v.z(), m(2, 1) * v.z(), m(2, 2) * v.z(), m(2, 3) * v.z());
	if ( colMajor ) m.transpose();
	return m;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::apply(const Vector4<Real>& v, bool colMajor) const {
	Matrix4<Real> m = transposed();
	m.setRow(0, m(0, 0) * v.w(), m(0, 1) * v.w(), m(0, 2) * v.w(), m(0, 3) * v.w());
	m.setRow(1, m(1, 0) * v.x(), m(1, 1) * v.x(), m(1, 2) * v.x(), m(1, 3) * v.x());
	m.setRow(2, m(2, 0) * v.y(), m(2, 1) * v.y(), m(2, 2) * v.y(), m(2, 3) * v.y());
	m.setRow(3, m(3, 0) * v.z(), m(3, 1) * v.z(), m(3, 2) * v.z(), m(3, 3) * v.z());
	if ( colMajor ) m.transpose();
	return m;
}

template <typename Real>
Vector3<Real> Matrix4<Real>::applyTo(const Vector3<Real>& v) const {
	Vector4<Real> result;
	result.x() = (*this)(0, 0) * v.x() + (*this)(0, 1) * v.y() + (*this)(0, 2) * v.z() + (*this)(0, 3);
	result.y() = (*this)(1, 0) * v.x() + (*this)(1, 1) * v.y() + (*this)(1, 2) * v.z() + (*this)(1, 3);
	result.z() = (*this)(2, 0) * v.x() + (*this)(2, 1) * v.y() + (*this)(2, 2) * v.z() + (*this)(2, 3);
	Real w   = (*this)(3, 0) * v.x() + (*this)(3, 1) * v.y() + (*this)(3, 2) * v.z() + (*this)(3, 3);

	if ( w != Real(0) && w != Real(1) )
		return Vector3<Real>(result.x() / w, result.y() / w, result.z() / w);
	return Vector3<Real>(result.x(), result.y(), result.z());
}

template <typename Real>
Vector4<Real> Matrix4<Real>::applyTo(const Vector4<Real>& v) const {
	return Vector4<Real>(
		(*this)(0, 0) * v.x() + (*this)(0, 1) * v.y() + (*this)(0, 2) * v.z() + (*this)(0, 3) * v.w(),
		(*this)(1, 0) * v.x() + (*this)(1, 1) * v.y() + (*this)(1, 2) * v.z() + (*this)(1, 3) * v.w(),
		(*this)(2, 0) * v.x() + (*this)(2, 1) * v.y() + (*this)(2, 2) * v.z() + (*this)(2, 3) * v.w(),
		(*this)(3, 0) * v.x() + (*this)(3, 1) * v.y() + (*this)(3, 2) * v.z() + (*this)(3, 3) * v.w()
	);
}

template <typename Real>
Matrix4<Real>::operator const Real* () const {
	return data;
}

template <typename Real>
Vector3<Real> operator*(const Matrix4<Real>& m, const Vector3<Real>& v) {
	return m.applyTo(v);
}

template <typename Real>
Vector4<Real> operator*(const Matrix4<Real>& m, const Vector4<Real>& v) {
	return m.applyTo(v);
}

template <typename Real>
std::ostream& operator<<(std::ostream& out, const Matrix4<Real>& m) {
	out << std::fixed << std::setprecision(6);
	out << "[" << m(0, 0) << "\t" << m(0, 1) << "\t" << m(0, 2) << "\t" << m(0, 3) << "]\n";
	out << "[" << m(1, 0) << "\t" << m(1, 1) << "\t" << m(1, 2) << "\t" << m(1, 3) << "]\n";
	out << "[" << m(2, 0) << "\t" << m(2, 1) << "\t" << m(2, 2) << "\t" << m(2, 3) << "]\n";
	out << "[" << m(3, 0) << "\t" << m(3, 1) << "\t" << m(3, 2) << "\t" << m(3, 3) << "]\n";
	return out;
}

template <typename Real>
bool Matrix4<Real>::operator==(const Matrix4<Real>& m) const {
    for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
		if ( data[i] != m.data[i] ) return false;
	}
	return true;
}

template <typename Real>
bool Matrix4<Real>::operator!=(const Matrix4<Real>& m) const {
	return !(*this == m);
}

template <typename Real>
Matrix4<Real>& Matrix4<Real>::operator=(const Matrix4<Real>& m) {
	if ( this != &m ) {
		std::memcpy(data, m.data, COMPONENT_COUNT * sizeof(Real));
	}
	return *this;
}

template <typename Real>
Matrix4<Real>& Matrix4<Real>::operator=(const Real* data) {
	if ( this->data != data ) {
		std::memcpy(this->data, data, COMPONENT_COUNT * sizeof(Real));
	}
	return *this;
}

template <typename Real>
Vector3<Real> Matrix4<Real>::operator*(const Vector3<Real>& v) const {
	return applyTo(v);
}

template <typename Real>
Vector4<Real> Matrix4<Real>::operator*(const Vector4<Real>& v) const {
	return applyTo(v);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::operator*(const Matrix4<Real>& m) const {
	return Multiply(*this, m);
}

template <typename Real>
Matrix4<Real>& Matrix4<Real>::operator*=(const Matrix4<Real>& m) {
	*this = Multiply(*this, m);
	return *this;
}

template <typename Real>
void Matrix4<Real>::ToRawMatrix(const Matrix4<Real>& m, Real* const matrix, bool colMajor) {
	m.toRawMatrix(matrix, colMajor);
}

template <typename Real>
void Matrix4<Real>::Clear(Matrix4<Real>& m) {
	m.zero();
}

template <typename Real>
void Matrix4<Real>::Identity(Matrix4<Real>& m) {
	m.identity();
}

template <typename Real>
void Matrix4<Real>::Zero(Matrix4<Real>& m) {
	m.zero();
}

template <typename Real>
Real Matrix4<Real>::Determinant(const Matrix4<Real>& m) {
	return det(m, nullptr);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Multiply(const Matrix4<Real>& a, const Matrix4<Real>& b) {
	Matrix4<Real> result(false);
	for ( int i = 0; i < ROW_COUNT; ++i ) {
		for ( int j = 0; j < COL_COUNT; ++j ) {
			for ( int k = 0; k < COL_COUNT; ++k ) {
				result(i, j) += a(i, k) * b(k, j);
			}
		}
	}
	return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Transpose(const Matrix4<Real>& m) {
	Matrix4<Real> result = m;
	result.transpose();
	return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Inverse(const Matrix4<Real>& matrix) {
	Real adjoint[COMPONENT_COUNT];
	Real det_val = det(matrix, adjoint);

	if ( std::abs(det_val) < 1e-6 )
		return Matrix4<Real>(true);

	Real invDet = Real(1) / det_val;
	for ( unsigned int i = 0; i < COMPONENT_COUNT; ++i ) {
		adjoint[i] = adjoint[i] * invDet;
	}
	return Matrix4<Real>(adjoint);
}

template <typename Real>
Real Matrix4<Real>::det(const Matrix4<Real>& matrix, Real* const adjoint_out) {
    // This is the raw unrolled determinant calculation.
	const Real* m = matrix.data;
	Real adjoint[16];

	adjoint[0]  = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
	adjoint[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
	adjoint[8]  = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
	adjoint[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
	adjoint[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
	adjoint[5]  = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
	adjoint[9]  = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
	adjoint[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
	adjoint[2]  = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
	adjoint[6]  = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
	adjoint[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1] *m[7] - m[12]*m[3]*m[5];
	adjoint[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
	adjoint[3]  = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
	adjoint[7]  = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
	adjoint[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
	adjoint[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    if ( adjoint_out != nullptr ) {
		std::memcpy(adjoint_out, adjoint, COMPONENT_COUNT * sizeof(Real));
	}

	return m[0]*adjoint[0] + m[1]*adjoint[4] + m[2]*adjoint[8] + m[3]*adjoint[12];
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::LookAt(const Vector3<Real>& eye, const Vector3<Real>& lookAt, const Vector3<Real>& up) {
	return LookAt(eye.x(), eye.y(), eye.z(), lookAt.x(), lookAt.y(), lookAt.z(), up.x(), up.y(), up.z());
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::LookAt(Real eyex, Real eyey, Real eyez, Real atx, Real aty, Real atz, Real upx, Real upy, Real upz) {
	Vector3<Real> eye(eyex, eyey, eyez);
	Vector3<Real> center(atx, aty, atz);
	Vector3<Real> up_vec(upx, upy, upz);

	Vector3<Real> f = (center - eye).normalized();
	Vector3<Real> s = Vector3<Real>::Cross(f, up_vec).normalized();
	Vector3<Real> u = Vector3<Real>::Cross(s, f);

	Matrix4<Real> result = Matrix4<Real>::Identity();
	result(0, 0) = s.x();
	result(0, 1) = s.y();
	result(0, 2) = s.z();
	result(1, 0) = u.x();
	result(1, 1) = u.y();
	result(1, 2) = u.z();
	result(2, 0) = -f.x();
	result(2, 1) = -f.y();
	result(2, 2) = -f.z();
	result(0, 3) = -Vector3<Real>::Dot(s, eye);
	result(1, 3) = -Vector3<Real>::Dot(u, eye);
	result(2, 3) = Vector3<Real>::Dot(f, eye);

	return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Perspective(Real fov, Real aspect, Real nearPlane, Real farPlane) {
	Matrix4<Real> result;

	Real angleRad = DegreesToRadians(fov);
    Real tanHalfFovy = std::tan(angleRad / Real(2));

    result(0,0) = Real(1) / (aspect * tanHalfFovy);
	result(1,0) = Real(0);
    result(2,0) = Real(0);
    result(3,0) = Real(0);

    result(0,1) = Real(0);
    result(1,1) = Real(1) / tanHalfFovy;
    result(2,1) = Real(0);
    result(3,1) = Real(0);

    result(0,2) = Real(0);
    result(1,2) = Real(0);
    result(2,2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
    result(3,2) = -Real(1);

    result(0,3) = Real(0);
    result(1,3) = Real(0);
    result(2,3) = -(Real(2) * farPlane * nearPlane) / (farPlane - nearPlane);
    result(3,3) = Real(0);

    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Zero() {
	return Matrix4<Real>(false);
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Identity() {
	return Matrix4<Real>(true);
}

/**
 * @brief Reduces a 4x4 matrix to a 3x3 matrix.
 * * This function extracts the upper-left 3x3 sub-matrix from a given 4x4 matrix.
 * This is particularly useful for operations like calculating a normal matrix, where only
 * the rotation and scaling components of a transformation are needed.
 *
 * @tparam Real The underlying numerical type (e.g., float, double) of the matrices.
 * @param matrix The constant reference to the source 4x4 matrix.
 * @return A new 3x3 matrix containing the top-left 3x3 elements of the source.
 */
template <typename Real>
Matrix3<Real> Matrix4<Real>::ToMatrix3(const Matrix4<Real>& matrix) {
    Matrix3<Real> result(false);
    for ( std::size_t row = 0; row < 3; ++row ) {
        for ( std::size_t col = 0; col < 3; ++col )
            result(row, col) = matrix(row, col);
    }
    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Translation(const Vector3<Real>& t) {
    Matrix4<Real> result(true);  // Start with identity
    result(0, 3) = t.x();
    result(1, 3) = t.y();
    result(2, 3) = t.z();
    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::Scale(const Vector3<Real>& s) {
    Matrix4<Real> result(true);  // Start with identity
    result(0, 0) = s.x();
    result(1, 1) = s.y();
    result(2, 2) = s.z();
    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::RotationX(Real angleRadians) {
    Matrix4<Real> result(true);
    Real c = std::cos(angleRadians);
    Real sn = std::sin(angleRadians);
    result(1, 1) = c;
    result(1, 2) = -sn;
    result(2, 1) = sn;
    result(2, 2) = c;
    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::RotationY(Real angleRadians) {
    Matrix4<Real> result(true);
    Real c = std::cos(angleRadians);
    Real sn = std::sin(angleRadians);
    result(0, 0) = c;
    result(0, 2) = sn;
    result(2, 0) = -sn;
    result(2, 2) = c;
    return result;
}

template <typename Real>
Matrix4<Real> Matrix4<Real>::RotationZ(Real angleRadians) {
    Matrix4<Real> result(true);
    Real c = std::cos(angleRadians);
    Real sn = std::sin(angleRadians);
    result(0, 0) = c;
    result(0, 1) = -sn;
    result(1, 0) = sn;
    result(1, 1) = c;
    return result;
}

typedef Matrix4<float> Matrix4f;
typedef Matrix4<double> Matrix4d;
typedef Matrix4<long> Matrix4l;
typedef Matrix4<int> Matrix4i;
typedef Matrix4<short> Matrix4s;
typedef Matrix4<float> Mat4f;
typedef Matrix4<double> Mat4d;
typedef Matrix4<long> Mat4l;
typedef Matrix4<int> Mat4i;
typedef Matrix4<short> Mat4s;
typedef Matrix4<float> Mat4;

}

#endif
