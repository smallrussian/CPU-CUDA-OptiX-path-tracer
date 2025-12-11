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
#ifndef COLOR_3_H
#define COLOR_3_H

#include <iostream>

namespace graphics {

template <typename Real> 
class Color3;

template <typename Real>
std::ostream& operator << (std::ostream& out, const Color3<Real>& color);

/* The Color3 class represents a 3-data color value (r, g, b). */
template <typename Real>
class Color3 {
    static_assert(std::is_integral<Real>::value || std::is_floating_point<Real>::value, 
    "[Color3:Type] Error: Color3 Real type must be an integral or floating point numerical representation.");

    enum Axis { R, G, B, COMPONENT_COUNT };

public:
	Color3(Real grayScale = 0);
	Color3(Real r, Real g, Real b);
	Color3(const Color3<Real>& color) = default;
	~Color3() = default;

	void setR(Real r);
	void setG(Real g);
	void setB(Real b);

	void addR(Real r);
	void addG(Real g);
	void addB(Real b);

	Real getR() const;
	Real getG() const;
	Real getB() const;
	Real& r();
	Real& g();
	Real& b();

    const Real& r() const;
    const Real& g() const;
    const Real& b() const;

	operator const Real*() const;
	operator Real*();

	bool operator == (const Color3<Real>& color);
	bool operator != (const Color3<Real>& color);

	// Arithmetic operators for ray tracing
	Color3<Real> operator + (const Color3<Real>& other) const;
	Color3<Real> operator * (const Color3<Real>& other) const;
	Color3<Real> operator * (Real scalar) const;

	friend std::ostream& operator << <> (std::ostream& out, const Color3<Real>& color);

	static Color3<Real> FromHSV(Real h, Real s, Real v);
	static Color3<Real> FromHSL(Real h, Real s, Real l);
    static Color3<Real> FromCMYK(Real c, Real m, Real y, Real k);

	void toHSV(Real& h, Real& s, Real& v) const;
    void toHSL(Real& h, Real& s, Real& l) const;
    void toCMYK(Real& c, Real& m, Real& y, Real& k) const;

protected:
	static Real hueToRGB(Real p, Real q, Real t);

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
Color3<Real>::Color3(Real grayScale) {
	this->data[R] = grayScale;
	this->data[G] = grayScale;
	this->data[B] = grayScale;
}

template <typename Real>
Color3<Real>::Color3(Real r, Real g, Real b) {
	this->data[R] = r;
	this->data[G] = g;
	this->data[B] = b;
}

template <typename Real>
void Color3<Real>::setR(Real r) {
	this->data[R] = r;
	clamp(this->data[R]);
}

template <typename Real>
void Color3<Real>::setG(Real g) {
	this->data[G] = g;
	clamp(this->data[G]);
}

template <typename Real>
void Color3<Real>::setB(Real b) {
	this->data[B] = b;
	clamp(this->data[B]);
}

template <typename Real>
void Color3<Real>::addR(Real r) {
	this->data[R] += r;
}

template <typename Real>
void Color3<Real>::addG(Real g) {
	this->data[G] += g;
}

template <typename Real>
void Color3<Real>::addB(Real b) {
	this->data[B] += b;
}

template <typename Real>
Real Color3<Real>::getR() const {
	return this->data[R];
}

template <typename Real>
Real Color3<Real>::getG() const {
	return this->data[G];
}

template <typename Real>
Real Color3<Real>::getB() const {
	return this->data[B];
}

template <typename Real>
Real& Color3<Real>::r() {
	return this->data[R];
}

template <typename Real>
Real& Color3<Real>::g() {
	return this->data[G];
}

template <typename Real>
Real& Color3<Real>::b() {
	return this->data[B];
}

template <typename Real>
const Real& Color3<Real>::r() const {
    return this->data[R];
}

template <typename Real>
const Real& Color3<Real>::g() const {
    return this->data[G];
}

template <typename Real>
const Real& Color3<Real>::b() const {
    return this->data[B];
}

template <typename Real>
Color3<Real>::operator const Real*() const {
	return &this->data[0];
}

template <typename Real>
Color3<Real>::operator Real*() {
	return &this->data[0];
}

template <typename Real>
bool Color3<Real>::operator == (const Color3<Real>& color) {
	if constexpr (std::is_floating_point_v<Real>) {
        return (std::abs(data[R] - color.data[R]) < std::numeric_limits<Real>::epsilon()) &&
               (std::abs(data[G] - color.data[G]) < std::numeric_limits<Real>::epsilon()) &&
               (std::abs(data[B] - color.data[B]) < std::numeric_limits<Real>::epsilon());
    } else {
        return data[R] == color.data[R] &&
               data[G] == color.data[G] &&
               data[B] == color.data[B];
    }

	if ( this->data[R] != color.data[R] ) return false;
	if ( this->data[G] != color.data[G] ) return false;
	if ( this->data[B] != color.data[B] ) return false;
	return true;
}

template <typename Real>
bool Color3<Real>::operator != (const Color3<Real>& color) {
	if ( *this == color ) return false;
	else return true;
}

template <typename Real>
std::ostream& operator << (std::ostream& out, const Color3<Real>& color) {
	out << "[ " << color.data[Color3<Real>::R] << " " << color.data[Color3<Real>::G] << " " << color.data[Color3<Real>::B] << " ]";
	return out;
}

template <typename Real>
Color3<Real> Color3<Real>::FromHSV(Real h, Real s, Real v) {
	unsigned int i = static_cast<unsigned int>(std::floor(h * 6));
    float f = h * 6 - static_cast<Real>(i);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1 - f) * s);
	float r, g, b;

    switch ( i % 6 ) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }

	Color3<Real> color;
    if constexpr (std::is_floating_point_v<Real>) {
        color.data[R] = static_cast<Real>(r);
        color.data[G] = static_cast<Real>(g);
        color.data[B] = static_cast<Real>(b);
    } else {
        color.data[R] = static_cast<Real>(r * 255.0f);
        color.data[G] = static_cast<Real>(g * 255.0f);
        color.data[B] = static_cast<Real>(b * 255.0f);
    }
    return color;
}

template <typename Real>
Color3<Real> Color3<Real>::FromHSL(Real h, Real s, Real l) {
    float r, g, b;

    if (s == 0.0f) {
        r = g = b = l; // achromatic
    } else {
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        r = hueToRGB(p, q, h + 1.0f/3.0f);
        g = hueToRGB(p, q, h);
        b = hueToRGB(p, q, h - 1.0f/3.0f);
    }

    Color3<Real> color;
    if constexpr (std::is_floating_point_v<Real>) {
        color.data[R] = static_cast<Real>(r);
        color.data[G] = static_cast<Real>(g);
        color.data[B] = static_cast<Real>(b);
    } else {
        color.data[R] = static_cast<Real>(r * 255.0f);
        color.data[G] = static_cast<Real>(g * 255.0f);
        color.data[B] = static_cast<Real>(b * 255.0f);
    }
    return color;
}

template <typename Real>
Real Color3<Real>::hueToRGB(Real p, Real q, Real t) {
    if ( t < Real(0) ) t += Real(1);
    if ( t > Real(1) ) t -= Real(1);
    if ( t < Real(1/6) ) return p + (q - p) * Real(6) * t;
    if ( t < Real(1/2) ) return q;
    if ( t < Real(2/3) ) return p + (q - p) * (Real(2/3) - t) * Real(6);
    return p;
}

template <typename Real>
Color3<Real> Color3<Real>::FromCMYK(Real c, Real m, Real y, Real k) {
    float r = (1.0f - c) * (1.0f - k);
    float g = (1.0f - m) * (1.0f - k);
    float b = (1.0f - y) * (1.0f - k);

    Color3<Real> color;
    if constexpr (std::is_floating_point_v<Real>) {
        color.data[R] = static_cast<Real>(r);
        color.data[G] = static_cast<Real>(g);
        color.data[B] = static_cast<Real>(b);
    } else {
        color.data[R] = static_cast<Real>(r * 255.0f);
        color.data[G] = static_cast<Real>(g * 255.0f);
        color.data[B] = static_cast<Real>(b * 255.0f);
    }
    return color;
}

template <typename Real>
void Color3<Real>::toHSV(Real& h, Real& s, Real& v) const {
    float r, g, b;
    if constexpr (std::is_floating_point_v<Real>) {
        r = static_cast<float>(this->data[R]);
        g = static_cast<float>(this->data[G]);
        b = static_cast<float>(this->data[B]);
    } else {
        r = static_cast<float>(this->data[R]) / 255.0f;
        g = static_cast<float>(this->data[G]) / 255.0f;
        b = static_cast<float>(this->data[B]) / 255.0f;
    }

    float cmax = std::max(std::max(r, g), b);
    float cmin = std::min(std::min(r, g), b);
    float delta = cmax - cmin;

    h = 0.0f;
    s = 0.0f;
    v = cmax;

    if ( cmax > 0.0f ) s = delta / cmax;

    if ( delta > 0.0f ) {
        if ( cmax == r ) h = (g - b) / delta;
        else if ( cmax == g ) h = (b - r) / delta + 2.0f;
        else h = (r - g) / delta + 4.0f;

        h /= 6.0f;

        if ( h < 0.0f )  h += 1.0f;
    }
}

template <typename Real>
void Color3<Real>::toHSL(Real& h, Real& s, Real& l) const {
	float r, g, b;
    if constexpr ( std::is_floating_point_v<Real> ) {
        r = static_cast<float>(this->data[R]);
        g = static_cast<float>(this->data[G]);
        b = static_cast<float>(this->data[B]);
    } else {
        r = static_cast<float>(this->data[R]) / 255.0f;
        g = static_cast<float>(this->data[G]) / 255.0f;
        b = static_cast<float>(this->data[B]) / 255.0f;
    }

    float cmax = std::max(std::max(r, g), b);
    float cmin = std::min(std::min(r, g), b);
    
    h = 0.0f;
	s = 0.0f,
	l = (cmax + cmin) / 2.0f;

    if ( cmax != cmin ) {
        float delta = cmax - cmin;
        s = l > 0.5f ? delta / (2.0f - cmax - cmin) : delta / (cmax + cmin);
        
        if ( cmax == r )  h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if ( cmax == g )  h = (b - r) / delta + 2.0f;
        else h = (r - g) / delta + 4.0f;

        h /= 6.0f;
    }
}

template <typename Real>
void Color3<Real>::toCMYK(Real& c, Real& m, Real& y, Real& k) const {
	float r, g, b;
    if constexpr (std::is_floating_point_v<Real>) {
        r = static_cast<float>(this->data[R]);
        g = static_cast<float>(this->data[G]);
        b = static_cast<float>(this->data[B]);
    } else {
        r = static_cast<float>(this->data[R]) / 255.0f;
        g = static_cast<float>(this->data[G]) / 255.0f;
        b = static_cast<float>(this->data[B]) / 255.0f;
    }

    if ( r == 0.0f && g == 0.0f && b == 0.0f ) {
		c = Real(0);
		m = Real(0);
		y = Real(0);
		k = Real(1);
	}
    
    k = 1.0f - std::max(std::max(r, g), b);
    c = (1.0f - r - k) / (1.0f - k);
    m = (1.0f - g - k) / (1.0f - k);
    y = (1.0f - b - k) / (1.0f - k);
}

// Arithmetic operator implementations
template <typename Real>
Color3<Real> Color3<Real>::operator + (const Color3<Real>& other) const {
    return Color3<Real>(data[R] + other.data[R],
                        data[G] + other.data[G],
                        data[B] + other.data[B]);
}

template <typename Real>
Color3<Real> Color3<Real>::operator * (const Color3<Real>& other) const {
    return Color3<Real>(data[R] * other.data[R],
                        data[G] * other.data[G],
                        data[B] * other.data[B]);
}

template <typename Real>
Color3<Real> Color3<Real>::operator * (Real scalar) const {
    return Color3<Real>(data[R] * scalar,
                        data[G] * scalar,
                        data[B] * scalar);
}

typedef Color3<float> Color3f;
typedef Color3<double> Color3d;
typedef Color3<int> Color3i;

}

#endif
