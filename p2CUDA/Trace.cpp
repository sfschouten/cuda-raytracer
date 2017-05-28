#include "Trace.h"

__device__ float3 PrimaryTrace::Do(Scene *scene, int recursionDepth)
{
	recursionDepth--;
	RayIntersection closest = scene->intersect(ray);
	Primitive *hit = closest.getPrimitive();

	Material hitMat = hit->getMaterial();
	float2 uv = closest.getTextureCoord();
	float3 diffuseColor = hitMat.getColor(uv.x, uv.y);
	Light *lights = scene->getLights();
	float3 totalLuminance;
	for (int i = 0; i < scene->getNrLights(); i++)
	{
		if (closest.getRay().length != FLT_MAX)
		{
			Light &l = lights[i];
			ShadowTrace st = ShadowTrace(closest, l);
			float3 luminance = st.Do(scene, recursionDepth);
			totalLuminance.x += luminance.x;
			totalLuminance.y += luminance.y;
			totalLuminance.z += luminance.z;
		}
		__syncthreads();
	}

	totalLuminance.x *= diffuseColor.x;
	totalLuminance.y *= diffuseColor.y;
	totalLuminance.z *= diffuseColor.z;

	return totalLuminance;
}

__device__ float3 ShadowTrace::Do(Scene *scene, int recursionDepth)
{
	float3 lColor = light.getColor();
	bool lit = !scene->shadowIntersect(ray);
	__syncthreads();
	float l = ray.length;
	float lightAttenuation = ((int)lit) / (l * l);
	float areaAttenuation = fmaxf(0.0f, Vector3::Dot(ri.getNormal(), ray.direction));
	//printf("l: %f, normal: (%f,%f,%f), direction: (%f,%f,%f)", l, ri.getNormal().x, ri.getNormal().y, ri.getNormal().z, ray.direction.x, ray.direction.y, ray.direction.z);
	lColor.x *= lightAttenuation * areaAttenuation;
	lColor.y *= lightAttenuation * areaAttenuation;
	lColor.z *= lightAttenuation * areaAttenuation;
	//printf("l: %f, a: %f \n", lightAttenuation, areaAttenuation);
	//printf("lColor: (%f, %f, %f) \n", lColor.x, lColor.y, lColor.z);
	return lColor;
}

__device__ Ray ShadowTrace::createShadowRay(RayIntersection& ri, Light& l)
{
	Ray oldRay = ri.getRay();
	Vector3 diff = l.getLocation() - oldRay.AsCoordinateVector();
	//printf("light: (%f,%f,%f) \n", l.getLocation().x, l.getLocation().y, l.getLocation().z);
	Ray newRay;
	newRay.direction = diff.normalized();
	//printf("diff: (%f,%f,%f) norm: (%f,%f,%f) \n", diff.x, diff.y, diff.z, newRay.direction.x, newRay.direction.y, newRay.direction.z);
	newRay.origin = newRay.direction * Epsilon + oldRay.AsCoordinateVector();
	newRay.length = diff.length() - Epsilon;
	return newRay;
}