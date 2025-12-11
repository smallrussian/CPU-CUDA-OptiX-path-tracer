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
#ifndef MESH_H
#define MESH_H

#include <vector>
#include "../math/Transformation.h"
#include "Shader.h"
#include "Color3.h"
#include "Vertex.h"
#include "Face.h"

namespace graphics {

class Mesh {
public:
    Mesh();
    virtual ~Mesh();

    bool load(const std::string& filename, bool bComputeNormals = false);
    bool setData(const std::vector<Vertex>& vertices, const std::vector<TriangleFace>& faces);
    bool loadShader(const std::string& vertexFilename, const std::string& fragmentFilename);

    void beginRender() const;
    void endRender() const;

    void setName(const std::string& name);
    void setShader(const std::shared_ptr<Shader>& shader);
    bool setDiffuseTexture(const std::string& filename);
    bool setNormalTexture(const std::string& filename);
    bool setSpecularTexture(const std::string& filename);

    void setPosition(float x, float y, float z);
    void setPosition(const Vector3f& position);
	void setScale(float s);
    void setScale(float sx, float sy, float sz);
    void setScale(const Vector3f& scale);
    void setRotation(const Quaternionf& rotation);

    std::string& getName();
    const std::string& getName() const;
    Transformationf& getTransform();
    const Transformationf& getTransform() const;
    std::shared_ptr<Shader>& getShader();
    const std::shared_ptr<Shader>& getShader() const;

    //ray tracing accessors - provide access to mesh geometry 
    const std::vector<Vertex>& getVertices() const;
    const std::vector<TriangleFace>& getFaces() const;

	std::string toString() const;

protected:
    bool constructOnGPU();

protected:
    /* 
     * Transformation that describes the position, scale, and rotation
     * of this mesh independent of its vertex positions.
     */
    Transformation<float> transform;

    std::string name;
    std::vector<Vertex> vertices;
    std::vector<TriangleFace> faces;
    std::shared_ptr<Shader> shader;

	unsigned int vao; // Vertex array object index (array)
    unsigned int vbo; // Vertex buffer object index (vertex data)
    unsigned int ebo; // Element buffer object index (indices)
};

}

#endif
