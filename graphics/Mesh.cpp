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
#define _CRT_SECURE_NO_WARNINGS
#include "Mesh.h"
#include <map>
#include <tuple>
#include <vector>
#include <iostream>
#include <sstream>

#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"

#include <QOpenGLFunctions_4_1_Core>
#include <QOpenGLVersionFunctionsFactory>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

namespace graphics {

const static unsigned int POSITION_LOC = 0;
const static unsigned int NORMAL_LOC = 1;
const static unsigned int TANGENT_LOC = 2;
const static unsigned int TEXTURE_COORD_LOC = 3;
const static unsigned int COLOR_LOC = 4;

Mesh::Mesh() {
    this->transform = Transformation<float>::Identity();
    this->shader = nullptr;
	this->vao = 0u;
	this->vbo = 0u;
	this->ebo = 0u;
}

Mesh::~Mesh() {
	// Only delete GL resources if we have a valid context
	auto context = QOpenGLContext::currentContext();
	if (!context) {
		// No GL context - can't delete GL resources (they may leak but won't crash)
		return;
	}

	auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(context);
	if (!f) return;

	if ( this->vao != 0u ) f->glDeleteVertexArrays(1, &this->vao);
	if ( this->vbo != 0u ) f->glDeleteBuffers(1, &this->vbo);
    if ( this->ebo != 0u ) f->glDeleteBuffers(1, &this->ebo);
}

/* http://www.terathon.com/code/tangent.html */
bool CalculateTangents(std::vector<Vertex>& vertices, std::vector<TriangleFace>& faces) {
	if ( vertices.size() == 0 ) {
		std::cerr << "[Mesh:calculateTangents] Error: Vertex array of length 0." << std::endl;
		return false;
	}

	if ( faces.size() == 0 ) {
		std::cerr << "[Mesh:calculateNormals] Error: Face count = 0." << std::endl;
		return false;
	}

	std::size_t triangleCount = faces.size();
	std::vector<Vector3f> tan1 = std::vector<Vector3f>(vertices.size());
	std::vector<Vector3f> tan2 = std::vector<Vector3f>(vertices.size());

	std::size_t i0 = 0, i1 = 0, i2 = 0;
	Vector3f p1, p2, p3;
	Vector3f w1, w2, w3;
	for ( std::size_t i = 0; i < triangleCount; i++ ) {
		i0 = faces[i].indices[A];
		i1 = faces[i].indices[B];
		i2 = faces[i].indices[C];

		p1 = vertices[i0].position;
		p2 = vertices[i1].position;
		p3 = vertices[i2].position;

		w1 = vertices[i0].textureCoord;
		w2 = vertices[i1].textureCoord;
		w3 = vertices[i2].textureCoord;

		float x1 = p2.x() - p1.x();
		float x2 = p3.x() - p1.x();
		float y1 = p2.y() - p1.y();
		float y2 = p3.y() - p1.y();
		float z1 = p2.z() - p1.z();
		float z2 = p3.z() - p1.z();

		float s1 = w2.x() - w1.x();
		float s2 = w3.x() - w1.x();
		float t1 = w2.y() - w1.y();
		float t2 = w3.y() - w1.y();

		float r = 1.0f / (s1 * t2 - s2 * t1);
		Vector3f sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
		Vector3f tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);

		tan1[i0] += sdir;
		tan1[i1] += sdir;
		tan1[i2] += sdir;

		tan2[i0] += tdir;
		tan2[i1] += tdir;
		tan2[i2] += tdir;
	}

	for ( std::size_t i = 0; i < vertices.size(); i++ ) {
		const Vector3f& n = vertices[i].normal;
		const Vector3f& t = tan1[i];

		vertices[i].tangent = Vector4f((t - n * (float)Vector3f::Dot(n, t)).normalized());
		if ( vertices[i].tangent.isEquivalent(Vector3f::Zero(), 0.01f) ) {
			if ( n.isEquivalent(Vector3f::UnitY(), 1.0e-4f) )
				vertices[i].tangent = Vector3f::UnitX();
			if ( n.isEquivalent(Vector3f::UnitNY(), 1.0e-4f) )
				vertices[i].tangent = Vector3f::UnitX();
		}

		if ( Vector3f::Dot(Vector3f::Cross(n, t), tan2[i]) < 0.0f ) vertices[i].tangent.w() = -1.0f;
		else vertices[i].tangent.w() = 1.0f;
	}

	return true;
}

bool CalculateNormals(
    const std::vector<unsigned int>& indices,
    const std::vector<Vector3f>& vertices,
    std::vector<Vector3f>& normals
    ) {

    if ( vertices.size() == 0 ) {
		std::cerr << "[Mesh:calculateNormals] Error: Vertex array of length 0." << std::endl;
		return false;
	}

	if ( indices.size() == 0 ) {
		std::cerr << "[Mesh:calculateNormals] Error: Face count = 0." << std::endl;
		return false;
	}

	std::vector<unsigned int> normalCounts = std::vector<unsigned int>(vertices.size());
    normals = std::vector<Vector3f>(vertices.size());
	std::size_t triangleCount = indices.size() / TRIANGLE_EDGE_COUNT;

	std::size_t i0 = 0, i1 = 0, i2 = 0;
	Vector3f p0, p1, p2, a, b, faceNormal;
    unsigned int index = 0;
	for ( std::size_t i = 0; i < triangleCount; i++) {
		i0 = indices[index];
        i1 = indices[index + 1];
        i2 = indices[index + 2];

		p0 = vertices[i0];
		p1 = vertices[i1];
		p2 = vertices[i2];

		a = p1 - p0;
		b = p2 - p0;

		faceNormal = a.cross(b);
		faceNormal.normalize();

        normals[i0] += faceNormal;
        normals[i1] += faceNormal;
        normals[i2] += faceNormal;

		normalCounts[i0] += 1;
		normalCounts[i1] += 1;
		normalCounts[i2] += 1;

        index += TRIANGLE_EDGE_COUNT;
	}

	for ( unsigned int i = 0; i < vertices.size(); i++ ) {
        normals[i] = normals[i] / static_cast<float>(normalCounts[i]);
        normals[i].normalize();
	}

	return true;
}

static void file_reader_callback(void* ctx, const char* filename, int is_mtl, const char* obj_filename, char** buf, size_t* len) {
    (void)ctx;
    (void)is_mtl;
    (void)obj_filename;

    long file_size = -1;
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        *buf = NULL;
        *len = 0;
        return;
    }

    fseek(fp, 0, SEEK_END);
    file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size < 0) {
        fclose(fp);
        *buf = NULL;
        *len = 0;
        return;
    }

    *buf = (char*)TINYOBJ_MALLOC(file_size);
    if (!(*buf)) {
        fclose(fp);
        *len = 0;
        return;
    }

    *len = fread(*buf, 1, file_size, fp);
    fclose(fp);
}

bool Mesh::load(const std::string& filename, bool bComputeNormals) {
    tinyobj_attrib_t attrib;
    tinyobj_shape_t* shapes = nullptr;
    size_t num_shapes;
    tinyobj_material_t* materials = nullptr;
    size_t num_materials;

    tinyobj_attrib_init(&attrib);

    int ret = tinyobj_parse_obj(
		&attrib,
		&shapes,
		&num_shapes,
		&materials,
		&num_materials,
        filename.c_str(),
		file_reader_callback,
		nullptr,
		TINYOBJ_FLAG_TRIANGULATE
	);

    if ( ret != TINYOBJ_SUCCESS ) {
        std::cerr << "[Mesh::load] Error: Failed to load/parse OBJ file: " << filename << std::endl;
        tinyobj_attrib_free(&attrib);
        tinyobj_shapes_free(shapes, num_shapes);
        tinyobj_materials_free(materials, num_materials);
        return false;
    }

    this->vertices.clear();
    this->faces.clear();

    std::map<std::tuple<int, int, int>, unsigned int> unique_vertices;

    for ( size_t i = 0; i < attrib.num_faces; i += 3 ) {
        TriangleFace face;
        for (int j = 0; j < 3; ++j) {
            tinyobj_vertex_index_t idx = attrib.faces[i + j];
            auto key = std::make_tuple(idx.v_idx, idx.vn_idx, idx.vt_idx);

            if ( unique_vertices.count(key) ) {
                face.indices[j] = unique_vertices[key];
            } else {
                Vertex v;

                v.position.set(
                    attrib.vertices[3 * (size_t)idx.v_idx + 0],
                    attrib.vertices[3 * (size_t)idx.v_idx + 1],
                    attrib.vertices[3 * (size_t)idx.v_idx + 2]
                );

                if (idx.vn_idx >= 0) {
                    v.normal.set(
                        attrib.normals[3 * (size_t)idx.vn_idx + 0],
                        attrib.normals[3 * (size_t)idx.vn_idx + 1],
                        attrib.normals[3 * (size_t)idx.vn_idx + 2]
                    );
                }

                if (idx.vt_idx >= 0) {
                    v.textureCoord.set(
                        attrib.texcoords[2 * (size_t)idx.vt_idx + 0],
                        attrib.texcoords[2 * (size_t)idx.vt_idx + 1],
                        0.0f
                    );
                }

                this->vertices.push_back(v);
                unsigned int new_index = static_cast<unsigned int>(this->vertices.size() - 1);
                unique_vertices[key] = new_index;
                face.indices[j] = new_index;
            }
        }
        this->faces.push_back(face);
    }
    
    bool hasNormalsInFile = (attrib.num_normals > 0);

    tinyobj_attrib_free(&attrib);
    tinyobj_shapes_free(shapes, num_shapes);
    tinyobj_materials_free(materials, num_materials);

    if ( bComputeNormals || !hasNormalsInFile ) {
        if ( !this->vertices.empty() && !this->faces.empty() ) {
            std::vector<unsigned int> flat_indices;
            std::vector<Vector3f> flat_positions;
            std::vector<Vector3f> calculated_normals;

            flat_indices.reserve(this->faces.size() * 3);
            for ( const auto& f : this->faces ) {
                flat_indices.push_back(f.indices[A]);
                flat_indices.push_back(f.indices[B]);
                flat_indices.push_back(f.indices[C]);
            }

            flat_positions.reserve(this->vertices.size());
            for ( const auto& v : this->vertices ) {
                flat_positions.push_back(v.position);
            }

            CalculateNormals(flat_indices, flat_positions, calculated_normals);
            
            for ( size_t i = 0; i < this->vertices.size(); ++i ) {
                this->vertices[i].normal = calculated_normals[i];
            }
        }
    }

    if ( !this->vertices.empty() && !this->faces.empty() ) {
        CalculateTangents(this->vertices, this->faces);
    }
    
	for ( unsigned int i = 0; i < this->vertices.size(); i++ ) {
		this->vertices[i].color = Color3f(0.0f, 0.0f, 0.0f);
    }

	this->constructOnGPU();
	return true;
}

bool Mesh::setData(const std::vector<Vertex>& verts, const std::vector<TriangleFace>& fcs) {
    this->vertices = verts;
    this->faces = fcs;

    if (this->vertices.empty() || this->faces.empty()) {
        std::cerr << "[Mesh::setData] Error: Empty vertex or face data" << std::endl;
        return false;
    }

    // Calculate tangents for normal mapping support
    CalculateTangents(this->vertices, this->faces);

    // Set default colors if not set
    for (auto& v : this->vertices) {
        if (v.color.r() == 0.0f && v.color.g() == 0.0f && v.color.b() == 0.0f) {
            v.color = Color3f(1.0f, 1.0f, 1.0f);
        }
    }

    this->constructOnGPU();
    return true;
}

bool Mesh::loadShader(const std::string& vertexFilename, const std::string& fragmentFilename) {
    this->shader = std::make_shared<Shader>();

    if ( !shader->load(vertexFilename, fragmentFilename) ) {
        std::cerr << "[Mesh:loadShader] Error: Could not load shader." << std::endl;
        return false;
    }

    if ( !shader->compile() ) {
        std::cerr << "[Mesh:loadShader] Error: Could not compile shader." << std::endl;
        return false;
    }

    if ( !shader->link() ) {
        std::cerr << "[Mesh:loadShader] Error: Could not link shader program." << std::endl;
        return false;
    }

    return true;
}

void Mesh::beginRender() const {
	if ( this->shader ) this->shader->enable();
}

void Mesh::endRender() const {
	auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

	f->glBindVertexArray(this->vao);
	
	f->glDrawElements(
		GL_TRIANGLES, 
		static_cast<GLsizei>(this->faces.size() * TRIANGLE_EDGE_COUNT), 
		GL_UNSIGNED_INT, 
		0
	);
	
	f->glBindVertexArray(0);

    if ( this->shader ) this->shader->disable();
}

void Mesh::setName(const std::string& name) {
    this->name = name;
}

void Mesh::setShader(const std::shared_ptr<Shader>& shader) {
    this->shader = shader;
}

bool Mesh::setDiffuseTexture(const std::string& filename) {
    if ( this->shader == nullptr ) return false;
    return this->shader->loadDiffuseTexture(filename);
}

bool Mesh::setNormalTexture(const std::string& filename) {
    if ( this->shader == nullptr ) return false;
    return this->shader->loadNormalTexture(filename);
}

bool Mesh::setSpecularTexture(const std::string& filename) {
    if ( this->shader == nullptr ) return false;
    return this->shader->loadSpecularTexture(filename);
}

void Mesh::setPosition(float x, float y, float z) {
    this->transform.setPosition(x, y, z);
}

void Mesh::setPosition(const Vector3f& position) {
    this->transform.setPosition(position);
}

void Mesh::setScale(float s) {
	this->transform.setScale(s, s, s);
}

void Mesh::setScale(float sx, float sy, float sz) {
    this->transform.setScale(sx, sy, sz);
}

void Mesh::setScale(const Vector3f& scale) {
    this->transform.setScale(scale);
}

void Mesh::setRotation(const Quaternionf& rotation) {
    this->transform.setRotation(rotation);
}

std::string& Mesh::getName() {
    return this->name;
}

const std::string& Mesh::getName() const {
    return this->name;
}

Transformationf& Mesh::getTransform() {
    return this->transform;
}

const Transformationf& Mesh::getTransform() const {
    return this->transform;
}

std::shared_ptr<Shader>& Mesh::getShader() {
    return this->shader;
}

const std::shared_ptr<Shader>& Mesh::getShader() const {
    return this->shader;
}

const std::vector<Vertex>& Mesh::getVertices() const {
    return this->vertices;
}

const std::vector<TriangleFace>& Mesh::getFaces() const {
    return this->faces;
}

std::string Mesh::toString() const {
	std::stringstream stream;
	std::size_t n = this->vertices.size();
	stream << "vertices = {" << std::endl;
	for ( std::size_t i = 0; i < n; ++i ) {
		stream << " vertex (" << i << ") {" << std::endl;
		stream << "  position   [" << this->vertices[i].position.x() << ',' << this->vertices[i].position.y() << ',' << this->vertices[i].position.z() << ']' << std::endl;
		stream << "  normal   [" << this->vertices[i].normal.x() << ',' << this->vertices[i].normal.y() << ',' << this->vertices[i].normal.z() << ']' << std::endl;
		stream << "  tangent  [" << this->vertices[i].tangent.x() << ',' << this->vertices[i].tangent.y() << ',' << this->vertices[i].tangent.z() << ',' << this->vertices[i].tangent.w() << ']' << std::endl;
		stream << "  texcoord [" << this->vertices[i].textureCoord.x() << ',' << this->vertices[i].textureCoord.y() << ',' << this->vertices[i].textureCoord.z() << ']' << std::endl;
		stream << "  color    [" << this->vertices[i].color.r() << ',' << this->vertices[i].color.g() << ',' << this->vertices[i].color.b() << ']' << std::endl;
		stream << " }" << std::endl;
	}
	stream << "}" << std::endl << std::endl;

	stream << "faces = {" << std::endl;
	for ( std::size_t i = 0; i < this->faces.size(); ++i ) {
		auto f = this->faces[i];
		stream << "  [" << f.indices[0] << ',' << f.indices[1] << ',' << f.indices[2] << ']' << std::endl;
	}
	stream << "}" << std::endl;

	return stream.str();
}

bool Mesh::constructOnGPU() {
	auto f = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_4_1_Core>(QOpenGLContext::currentContext());

	f->glGenVertexArrays(1, &this->vao);
	f->glGenBuffers(1, &this->vbo);
	f->glGenBuffers(1, &this->ebo);

	f->glBindVertexArray(this->vao);

	// Vertex data
    f->glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
	f->glBufferData(GL_ARRAY_BUFFER, this->vertices.size() * sizeof(Vertex), &this->vertices[0], GL_STATIC_DRAW);

	// Vertex array data attributes
	f->glEnableVertexAttribArray(POSITION_LOC);
	f->glVertexAttribPointer(POSITION_LOC, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(0));

	f->glEnableVertexAttribArray(NORMAL_LOC);
	f->glVertexAttribPointer(NORMAL_LOC, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(3 * sizeof(float)));

	f->glEnableVertexAttribArray(TANGENT_LOC);
	f->glVertexAttribPointer(TANGENT_LOC, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(6 * sizeof(float)));

	f->glEnableVertexAttribArray(TEXTURE_COORD_LOC);
	f->glVertexAttribPointer(TEXTURE_COORD_LOC, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(10 * sizeof(float)));

	f->glEnableVertexAttribArray(COLOR_LOC);
	f->glVertexAttribPointer(COLOR_LOC, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(13 * sizeof(float)));
	
	// Element indices
    f->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->ebo);
    f->glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->faces.size() * TRIANGLE_EDGE_COUNT * sizeof(unsigned int), &this->faces[0].indices[0], GL_STATIC_DRAW);
    
	f->glBindVertexArray(0);
    return true;
}

}
