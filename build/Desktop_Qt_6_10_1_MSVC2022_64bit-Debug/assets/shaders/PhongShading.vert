#version 410 core

/* Uniform variables for Camera */
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

/* Light properties */
uniform vec3 lightPosition;   // Point/sphere light position
uniform vec3 lightDirection;  // Directional light direction

/* Vertex attributes */
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec3 textureCoordinate;
layout(location = 4) in vec3 color;

/*
 * Light model information interpolated between each vertex. This information
 * is used to compute the light model within the fragment shader based on the
 * interpolated output values.
 */
out vec3 interp_surfaceNormal;
out vec3 interp_vertexPosition;
out vec3 interp_lightPosition;
out vec3 interp_lightDirection;

/* Phong Shading */
void main(void) {
	// Calculate vertex and light position in view space
	interp_lightPosition = vec3(modelViewMatrix * vec4(lightPosition, 1.0f));
	interp_vertexPosition = vec3(modelViewMatrix * vec4(position, 1.0f));

	// Transform the light direction into view space (direction, not position - use 0.0 for w)
	interp_lightDirection = mat3(modelViewMatrix) * lightDirection;

	// Transform the normal into view space
	interp_surfaceNormal = normalize(mat3(normalMatrix) * normal);

	// Transform the vertex for the fragment shader.
	gl_Position = projectionMatrix * vec4(interp_vertexPosition, 1.0f);
}
