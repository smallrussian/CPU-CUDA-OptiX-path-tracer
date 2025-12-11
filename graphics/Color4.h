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
#ifndef COLOR_4_H
#define COLOR_4_H

#include <iostream>

namespace graphics {

template <typename Real> 
class Color4;

template <typename Real>
std::ostream& operator << (std::ostream& out, const Color4<Real>& color);

/* The Color4 class represents a 4-data color value (r, g, b, a). */
template <typename Real>
class Color4 {
    static_assert(std::is_integral<Real>::value || std::is_floating_point<Real>::value, 
    "[Color4:Type] Error: Color4 Real type must be an integral or floating point numerical representation.");

    enum Axis { R, G, B, A, COMPONENT_COUNT };

public:
	Color4(Real grayScale = 0);
	Color4(Real r, Real g, Real b, Real a);
	Color4(const Color4<Real>& color) = default;
	~Color4() = default;

	void setR(Real r);
	void setG(Real g);
	void setB(Real b);
	void setA(Real a);

	void addR(Real r);
	void addG(Real g);
	void addB(Real b);
	void addA(Real a);

	Real getR() const;
	Real getG() const;
	Real getB() const;
	Real getA() const;

	Real& r();
	Real& g();
	Real& b();
	Real& a();

    const Real& r() const;
    const Real& g() const;
    const Real& b() const;
    const Real& a() const;

	operator const Real*() const;
	operator Real* ();

	bool operator == (const Color4<Real>& color);
	bool operator != (const Color4<Real>& color);

    friend std::ostream& operator << <> (std::ostream& out, const Color4<Real>& color);

private:
    static void clamp(Real& val) {
        if constexpr (std::is_floating_point_v<Real>) {
            if (val < Real(0)) val = Real(0);
            if (val > Real(1)) val = Real(1);
        } else {
            if (val < 0) val = 0;
            if (val > 255) val = 255;
        }
    }

	Real data[COMPONENT_COUNT];
};

template <typename Real>
Color4<Real>::Color4(Real grayScale) {
	this->data[R] = grayScale;
	this->data[G] = grayScale;
	this->data[B] = grayScale;
	this->data[A] = grayScale;
}

template <typename Real>
Color4<Real>::Color4(Real r, Real g, Real b, Real a) {
	this->data[R] = r;
	this->data[G] = g;
	this->data[B] = b;
	this->data[A] = a;
}

template <typename Real>
void Color4<Real>::setR(Real r) {
	this->data[R] = r;
	clamp(this->data[R]);
}

template <typename Real>
void Color4<Real>::setG(Real g) {
	this->data[G] = g;
	clamp(this->data[G]);
}

template <typename Real>
void Color4<Real>::setB(Real b) {
	this->data[B] = b;
	clamp(this->data[B]);
}

template <typename Real>
void Color4<Real>::setA(Real a) {
	this->data[A] = a;
	clamp(this->data[A]);
}

template <typename Real>
void Color4<Real>::addR(Real r) {
	this->data[R] += r;
}

template <typename Real>
void Color4<Real>::addG(Real g) {
	this->data[G] += g;
}

template <typename Real>
void Color4<Real>::addB(Real b) {
	this->data[B] += b;
}

template <typename Real>
void Color4<Real>::addA(Real a) {
	this->data[A] += a;
}

template <typename Real>
Real Color4<Real>::getR() const {
	return this->data[R];
}

template <typename Real>
Real Color4<Real>::getG() const {
	return this->data[G];
}

template <typename Real>
Real Color4<Real>::getB() const {
	return this->data[B];
}

template <typename Real>
Real Color4<Real>::getA() const {
	return this->data[A];
}

template <typename Real>
Real& Color4<Real>::r() {
	return this->data[R];
}

template <typename Real>
Real& Color4<Real>::g() {
	return this->data[G];
}

template <typename Real>
Real& Color4<Real>::b() {
	return this->data[B];
}

template <typename Real>
Real& Color4<Real>::a() {
	return this->data[A];
}

template <typename Real>
const Real& Color4<Real>::r() const {
    return this->data[R];
}

template <typename Real>
const Real& Color4<Real>::g() const {
    return this->data[G];
}

template <typename Real>
const Real& Color4<Real>::b() const {
    return this->data[B];
}

template <typename Real>
const Real& Color4<Real>::a() const {
    return this->data[A];
}

template <typename Real>
Color4<Real>::operator const Real*() const {
	return &this->data[0];
}

template <typename Real>
Color4<Real>::operator Real* () {
	return &this->data[0];
}

template <typename Real>
bool Color4<Real>::operator == (const Color4<Real>& color) {
	if ( this->data[R] != color.data[R] ) return false;
	if ( this->data[G] != color.data[G] ) return false;
	if ( this->data[B] != color.data[B] ) return false;
	if ( this->data[A] != color.data[A] ) return false;
	return true;
}

template <typename Real>
bool Color4<Real>::operator != (const Color4<Real>& color) {
	if ( *this == color ) return false;
	else return true;
}

template <typename Real>
std::ostream& operator << (std::ostream& out, const Color4<Real>& color) {
	out << "[ " << color.data[Color4<Real>::R] << " " << color.data[Color4<Real>::G] << " " << color.data[Color4<Real>::B] << " " << color.data[Color4<Real>::A] << " ]";
	return out;
}

typedef Color4<float> Color4f;
typedef Color4<double> Color4d;
typedef Color4<int> Color4i;

}

#endif
