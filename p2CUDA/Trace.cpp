#include "Trace.h"

__device__ float3 PrimaryTrace::Do(Scene *scene, int recursionDepth)
{
	RayIntersection closest = scene->intersect(ray);
	Primitive *hit = closest.getPrimitive();

	float3 result = make_float3(0, 0, 0);
	if (closest.getRay().length != FLT_MAX)
	{
		Material hitMat = hit->getMaterial();
		float2 uv = closest.getTextureCoord();
		float3 diffuseColor = hitMat.getDiffuseColor(uv.x, uv.y);
		Light *lights = scene->getLights();
		
		for (int i = 0; i < scene->getNrLights(); i++)
		{
			Light &l = lights[i];
			ShadowTrace st = ShadowTrace(closest, l);
			float3 luminance = st.Do(scene, -1); //recursionDepth doesn't matter here.
			result.x += luminance.x;
			result.y += luminance.y;
			result.z += luminance.z;
		}

		result.x *= diffuseColor.x;
		result.y *= diffuseColor.y;
		result.z *= diffuseColor.z;

		if (recursionDepth > 0)
		{
			Vector3 d = closest.getRay().direction;
			Vector3 newV = d - closest.getNormal() * Vector3::Dot(closest.getNormal(), d) * 2;
			Ray bounced = Ray(closest.getRay().AsCoordinateVector() + newV * Epsilon, newV.normalized());
			PrimaryTrace subTrace = PrimaryTrace(bounced);
			float3 subColor = subTrace.Do(scene, recursionDepth - 1);
			float3 specColor = hitMat.getSpecularColor(uv.x, uv.y);
			result.x += specColor.x * subColor.x;
			result.y += specColor.y * subColor.y;
			result.z += specColor.z * subColor.z;
		}
	}

	__syncthreads();
	return result;
}

__device__ float3 ShadowTrace::Do(Scene *scene, int recursionDepth)
{
	float3 lColor = light.getDiffuseColor();
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