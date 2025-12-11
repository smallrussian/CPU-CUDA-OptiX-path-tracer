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
#ifndef CAMERA_H
#define CAMERA_H

#include "../math/Mathematics.h"
#include "../math/Matrix4.h"

namespace graphics {

template <typename Real>
class Camera {
public:
    Camera(Real radius = Real(1));
    Camera(const Camera<Real>& camera);
    ~Camera() = default;

    void rotate(Real dTheta, Real dPhi);

    void setPerspective(Real fov, Real aspectRatio, Real znear, Real zfar);
    void setFrustum(Real left, Real right, Real top, Real bottom, Real znear, Real zfar);

    void setRotation(Real theta, Real phi);
    void setPosition(Real x, Real y, Real z);
    void setPosition(const Vector3<Real>& position);
    void setRadius(Real radius);
    void addRadius(Real dRadius);
	void setFOV(Real fov);
	void setNearPlane(Real d);
	void setFarPlane(Real d);

	Real getNearPlane() const;
	Real getFarPlane() const;
	Real getTheta() const;
	Real getPhi() const;
	Real getFOV() const;

    Matrix3<Real> getBasisMatrix(bool colMajor = true) const;
    Matrix4<Real> toViewMatrix() const;
    Matrix4<Real> toProjectionMatrix() const;
    Vector3<Real> toCartesianCoordinates() const;
    Vector3<Real> toSphericalCoordinates() const;

    Vector3<Real> getEyeDirection() const;
    Vector3<Real> getUpDirection() const;
    Vector3<Real> getRightDirection() const;

    Matrix4<Real>& getViewMatrix();
    Matrix4<Real>& getProjectionMatrix();
    Real& getRadius();
    Vector3<Real>& getEye();
    Vector3<Real>& getLookAt();
    Vector3<Real>& getUp();
    Vector3<Real>& getRight();

    const Matrix4<Real>& getViewMatrix() const;
    const Matrix4<Real>& getProjectionMatrix() const;

    const Real& getRadius() const;
    const Vector3<Real>& getEye() const;
    const Vector3<Real>& getLookAt() const;
    const Vector3<Real>& getUp() const;
    const Vector3<Real>& getRight() const;

protected:
    void compile();

private:
    Matrix4<Real> view;
    Matrix4<Real> projection;
	Real fov;
	Real clip_near;
	Real clip_far;

    Vector3<Real> eye;
    Vector3<Real> lookAt;

    Vector3<Real> up;
    Vector3<Real> right;
    Vector3<Real> dir;

    Real r, theta, phi;
};

template <typename Real>
Camera<Real>::Camera(Real radius) {
    this->r = radius;
    this->theta = Real(0);
    this->phi = Real(HALF_PI);
	this->fov = Real(45);
	this->clip_near = Real(0.1);
	this->clip_far = Real(100);

    this->eye = Vector3<Real>::UnitZ();
    this->lookAt = Vector3<Real>::Zero();
    this->up = Vector3<Real>::UnitY();

    this->compile();
}

template <typename Real>
Camera<Real>::Camera(const Camera<Real>& camera) {
    this->view = camera.view;
    this->eye = camera.eye;
    this->lookAt = camera.lookAt;
    this->up = camera.up;
    this->r = camera.r;
    this->theta = camera.theta;
    this->phi = camera.phi;
	this->right = camera.right;
	this->dir = camera.dir;
	this->fov = camera.fov;
	this->clip_far = camera.clip_far;
	this->clip_near = camera.clip_near;
	this->projection = camera.projection;
    // this->compile();
}

template <typename Real>
void Camera<Real>::rotate(Real dTheta, Real dPhi) {
    this->theta += dTheta;
    this->phi += dPhi;
    this->compile();
}

template <typename Real>
void Camera<Real>::setPerspective(Real fov, Real aspectRatio, Real znear, Real zfar) {
    Real angleRad = DegreesToRadians(fov);
    Real tanHalfFovy = std::tan(angleRad / Real(2));

    this->projection(0,0) = Real(1) / (aspectRatio * tanHalfFovy);
	this->projection(1,0) = Real(0);
    this->projection(2,0) = Real(0);
    this->projection(3,0) = Real(0);

    this->projection(0,1) = Real(0);
    this->projection(1,1) = Real(1) / tanHalfFovy;
    this->projection(2,1) = Real(0);
    this->projection(3,1) = Real(0);

    this->projection(0,2) = Real(0);
    this->projection(1,2) = Real(0);
    this->projection(2,2) = -(zfar + znear) / (zfar - znear);
    this->projection(3,2) = -Real(1);

    this->projection(0,3) = Real(0);
    this->projection(1,3) = Real(0);
    this->projection(2,3) = -(Real(2) * zfar * znear) / (zfar - znear);
    this->projection(3,3) = Real(0);
}

template <typename Real>
void Camera<Real>::setFrustum(Real left, Real right, Real top, Real bottom, Real znear, Real zfar) {
    this->projection.zero();

    Real z, width, height, d;
    z = Real(2) * znear;
    width = right - left;
    height = top - bottom;
    d = zfar - znear;

    this->projection[0] = z / width;
    this->projection[5] = z / height;
    this->projection[8] = (right + left) / width;
    this->projection[9] = (top + bottom) / height;
    this->projection[10] = (-zfar - znear) / d;
    this->projection[11] = Real(-1);
    this->projection[14] = (-z * zfar) / d;
}

template <typename Real>
void Camera<Real>::setRotation(Real theta, Real phi) {
    this->theta = theta;
    this->phi = phi;
    this->compile();
}

template <typename Real>
void Camera<Real>::setPosition(Real x, Real y, Real z) {
    this->lookAt.set(x, y, z);
    this->compile();
}

template <typename Real>
void Camera<Real>::setPosition(const Vector3<Real>& position) {
    this->lookAt = position;
    this->compile();
}

template <typename Real>
void Camera<Real>::setRadius(Real radius) {
    this->r = radius;
    this->compile();
}

template <typename Real>
void Camera<Real>::addRadius(Real dRadius) {
    this->r += dRadius;
    this->compile();
}

template <typename Real>
void Camera<Real>::setFOV(Real fov) {
	this->fov = fov;
}

template <typename Real>
void Camera<Real>::setNearPlane(Real d) {
	this->clip_near = d;
}

template <typename Real>
void Camera<Real>::setFarPlane(Real d) {
	this->clip_far = d;
}

template <typename Real>
Real Camera<Real>::getNearPlane() const {
	return this->clip_near;
}

template <typename Real>
Real Camera<Real>::getFarPlane() const {
	return this->clip_far;
}

template <typename Real>
Real Camera<Real>::getTheta() const {
	return this->theta;
}

template <typename Real>
Real Camera<Real>::getPhi() const {
	return this->phi;
}

template <typename Real>
Real Camera<Real>::getFOV() const {
	return this->fov;
}

template <typename Real>
Matrix3<Real> Camera<Real>::getBasisMatrix(bool colMajor) const {
    Matrix3<Real> matrix;

    if ( colMajor ) {
        matrix.setColumn(0, this->dir);
        matrix.setColumn(1, this->up);
        matrix.setColumn(2, this->right);
    }
    else {
        matrix.setRow(0, this->dir);
        matrix.setRow(1, this->up);
        matrix.setRow(2, this->right);
    }

    return matrix;
}

template <typename Real>
Matrix4<Real> Camera<Real>::toViewMatrix() const {
    this->compile();
    return this->view;
}

template <typename Real>
Matrix4<Real> Camera<Real>::toProjectionMatrix() const {
    return this->projection;
}

template <typename Real>
Vector3<Real> Camera<Real>::toCartesianCoordinates() const {
    return SphericalToCartesian<Real>(this->r, this->theta, this->phi);
}

template <typename Real>
Vector3<Real> Camera<Real>::toSphericalCoordinates() const {
    return Vector3<Real>(this->r, this->theta, this->phi);
}

template <typename Real>
Vector3<Real> Camera<Real>::getEyeDirection() const {
    return (this->lookAt - this->eye).normalize();
}

template <typename Real>
Vector3<Real> Camera<Real>::getUpDirection() const {
    return this->up;
}

template <typename Real>
Vector3<Real> Camera<Real>::getRightDirection() const {
    return this->right;
}

template <typename Real>
Matrix4<Real>& Camera<Real>::getViewMatrix() {
    return this->view;
}

template <typename Real>
Matrix4<Real>& Camera<Real>::getProjectionMatrix() {
    return this->projection;
}

template <typename Real>
Real& Camera<Real>::getRadius() {
    return this->r;
}

template <typename Real>
Vector3<Real>& Camera<Real>::getEye() {
    return this->eye;
}

template <typename Real>
Vector3<Real>& Camera<Real>::getLookAt() {
    return this->lookAt;
}

template <typename Real>
Vector3<Real>& Camera<Real>::getUp() {
    return this->up;
}

template <typename Real>
Vector3<Real>& Camera<Real>::getRight() {
    return this->right;
}

template <typename Real>
const Matrix4<Real>& Camera<Real>::getViewMatrix() const {
    return this->view;
}

template <typename Real>
const Matrix4<Real>& Camera<Real>::getProjectionMatrix() const {
    return this->projection;
}

template <typename Real>
const Real& Camera<Real>::getRadius() const {
    return this->r;
}

template <typename Real>
const Vector3<Real>& Camera<Real>::getEye() const {
    return this->eye;
}

template <typename Real>
const Vector3<Real>& Camera<Real>::getLookAt() const {
    return this->lookAt;
}

template <typename Real>
const Vector3<Real>& Camera<Real>::getUp() const {
    return this->up;
}

template <typename Real>
const Vector3<Real>& Camera<Real>::getRight() const {
    return this->right;
}

template <typename Real>
void Camera<Real>::compile() {
    this->eye = SphericalToCartesian<Real>(this->r, this->theta, this->phi);
    this->up = -SphericalToCartesian_dPhi<Real>(this->r, this->theta, this->phi);
    this->right = SphericalToCartesian_dTheta<Real>(this->r, this->theta, this->phi);
    this->dir = SphericalToCartesian_dPhiCrossdTheta(this->r, this->theta, this->phi);

    this->eye.swapYZ();
    this->right.swapYZ();
    this->up.swapYZ();
    this->dir.swapYZ();

    this->right.normalize();
    this->up.normalize();
    this->dir.normalize();

    this->view = Matrix4<Real>::LookAt(this->eye + this->lookAt, this->lookAt, this->up);
}

typedef Camera<float> Cameraf;
typedef Camera<double> Camerad;

}

#endif
