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
#ifndef SHADER_H
#define SHADER_H

#include <memory>
#include <string>
#include "../math/Matrix4.h"
#include "../graphics/Texture.h"
#include "../graphics/Color3.h"
#include "../graphics/Color4.h"

namespace graphics {

/* Wrapper class around the basic OpenGL functions that support GLSL shaders. */
class Shader {
public:
    Shader();
    Shader(const Shader& shader);
    ~Shader();

    bool load(const std::string& vertexFilename, const std::string& fragmentFilename);
    bool compile();
    bool link();

    bool loadDiffuseTexture(const std::string& filename);
    bool loadNormalTexture(const std::string& filename);
    bool loadSpecularTexture(const std::string& filename);

    bool enable();
    bool disable();

    operator unsigned int () const;
    unsigned int getProgramID() const;
    unsigned int id() const;

    void uniform1f(const std::string& name, float value) const;
	void uniform1fv(const std::string& name, unsigned int count, float* values) const;
	void uniform1d(const std::string& name, double value) const;
	void uniform1dv(const std::string& name, unsigned int count, double* values) const;
	void uniform2f(const std::string& name, float value0, float value1) const;
	void uniform2fv(const std::string& name, unsigned int count, float* values) const;
	void uniform2d(const std::string& name, double value0, double value1) const;
	void uniform2dv(const std::string& name, unsigned int count, double* values) const;
	void uniform3f(const std::string& name, float value0, float value1, float value2) const;
	void uniform3fv(const std::string& name, unsigned int count, float* values) const;
	void uniform3d(const std::string& name, double value0, double value1, double value2) const;
	void uniform3dv(const std::string& name, unsigned int count, double* values) const;
	void uniform4f(const std::string& name, float value0, float value1, float value2, float value3) const;
	void uniform4fv(const std::string& name, unsigned int count, float* values) const;
	void uniform4d(const std::string& name, double value0, double value1, double value2, double value3) const;
	void uniform4dv(const std::string& name, unsigned int count, double* values) const;
	void uniform1i(const std::string& name, int value) const;
	void uniform2i(const std::string& name, int value0, int value1) const;
	void uniform3i(const std::string& name, int value0, int value1, int value2) const;
	void uniform4i(const std::string& name, int value0, int value1, int value2, int value3) const;

    void uniformMatrix(const std::string& name, const Matrix4f& matrix) const;
    void uniformMatrix(const std::string& name, const Matrix3f& matrix) const;
    void uniformVector(const std::string& name, const Vector3f& vector) const;
    void uniformVector(const std::string& name, const Vector4f& vector) const;
	void uniformColor(const std::string& name, const Color3f& color) const;
	void uniformColor(const std::string& name, const Color4f& color) const;

protected:
    bool loadFile(const std::string& filename, std::string& content);
    bool compileStatus(unsigned int shaderId, const std::string& filename) const;
    bool linkStatus(unsigned int programId) const;

protected:
    unsigned int programId;
    unsigned int vertexId;
    unsigned int fragmentId;

    /* 2D Texture data of this mesh */
    std::shared_ptr<Texture> diffuseTexture;
    std::shared_ptr<Texture> normalTexture;
    std::shared_ptr<Texture> specularTexture;

    /* Shader File Info */
    std::string vertFilename;
    std::string fragFilename;
    std::string vertSource;
    std::string fragSource;
};

}

#endif
