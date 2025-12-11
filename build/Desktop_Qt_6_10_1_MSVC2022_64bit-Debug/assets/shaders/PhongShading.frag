#version 410 core

/*
 * Light model information interpolated between each vertex. This information is
 * used to compute the light model at each fragment based on the
 * interpolated values.
 */
in vec3 interp_surfaceNormal;
in vec3 interp_vertexPosition;
in vec3 interp_lightPosition;   // Point/sphere light position (in view space)
in vec3 interp_lightDirection;  // Directional light direction (in view space)

/* Material properties (passed from application) */
uniform vec4 Ka;  // Ambient reflectivity
uniform vec4 Kd;  // Diffuse reflectivity
uniform vec4 Ks;  // Specular reflectivity
uniform float shininess;

/* Light properties (passed from application) */
uniform vec4 Ia;  // Ambient light intensity
uniform vec4 Id;  // Diffuse light intensity (point/sphere light)
uniform vec4 Is;  // Specular light intensity
uniform vec4 IdDir;  // Directional light diffuse intensity
uniform int hasDirectionalLight;  // 1 if scene has a directional light
uniform int hasPointLight;        // 1 if scene has a point/sphere light

/* Output fragment */
out vec4 out_frag;

/* Phong Shading */
void main(void) {
	vec3 n = normalize(interp_surfaceNormal);
	vec3 c = normalize(-interp_vertexPosition);  // View direction

	vec4 Iambient = vec4(0.0f);
	vec4 Idiffuse = vec4(0.0f);
	vec4 Ispecular = vec4(0.0f);

	// Ambient component - always applied
	Iambient = Ia * Ka;

	// Directional light contribution (like sun)
	if (hasDirectionalLight == 1) {
		vec3 lDir = normalize(-interp_lightDirection);  // Light points opposite to direction
		vec3 rDir = normalize(-reflect(lDir, n));

		float lambertDir = max(0.0f, dot(n, lDir));
		Idiffuse += (IdDir * Kd) * lambertDir;

		float specDir = pow(max(dot(rDir, c), 0.0f), shininess);
		Ispecular += (Is * Ks) * specDir * 0.5f;  // Reduced specular for sun
	}

	// Point/sphere light contribution
	if (hasPointLight == 1) {
		vec3 lPoint = normalize(interp_lightPosition - interp_vertexPosition);
		vec3 rPoint = normalize(-reflect(lPoint, n));

		float lambertPoint = max(0.0f, dot(n, lPoint));
		Idiffuse += (Id * Kd) * lambertPoint;

		float specPoint = pow(max(dot(rPoint, c), 0.0f), shininess);
		Ispecular += (Is * Ks) * specPoint;
	}

	// Calculate the final ADS light value for this vertex.
	out_frag = Iambient + Idiffuse + Ispecular;
}
