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
#include "Texture.h"
#include <iostream>
#include <QImage>
#include <QOpenGLFunctions_4_1_Core>
#include <QOpenGLVersionFunctionsFactory>

namespace graphics {

Texture::Texture() {
    this->image_width = 0;
    this->image_height = 0;
    this->textureId = 0u;
}

Texture::~Texture() {
	this->release();
}

bool Texture::load(const std::string& filename) {
    if ( filename.length() == 0 ) return false;

    QImage img;
    if ( !img.load(filename.c_str()) ) {
        std::cerr << "[Texture::load] Error: Could not load image: "  << filename << std::endl;
        return false;
    }

    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

    QImage imgRGBA = img.convertToFormat(QImage::Format_RGBA8888).flipped(Qt::Vertical);

    this->image_width = imgRGBA.width();
    this->image_height = imgRGBA.height();

    // Generate and bind OpenGL texture
    f->glGenTextures(1, &this->textureId);
    f->glBindTexture(GL_TEXTURE_2D, this->textureId);

    // Set texture parameters
    f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload pixel data
    f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->image_width, this->image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imgRGBA.bits());
    f->glBindTexture(GL_TEXTURE_2D, 0);
    
    return true;
}

void Texture::render() const {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

    f->glEnable(GL_TEXTURE_2D);
    f->glBindTexture(GL_TEXTURE_2D, this->textureId);
}

void Texture::release() const {
    auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

    f->glDeleteTextures(1, &this->textureId);
}

unsigned int Texture::id() const {
	return this->textureId;
}

int Texture::getWidth() const {
	return this->image_width;
}

int Texture::getHeight() const {
	return this->image_height;
}

int Texture::width() const {
	return this->image_width;
}

int Texture::height() const {
	return this->image_height;
}

}
