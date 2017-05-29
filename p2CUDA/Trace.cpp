#include "Trace.h"

__device__ float3 PrimaryTrace::Do(Scene *scene, int recursionDepth)
{
	RayIntersection closest = scene->intersect(ray);
	Primitive *hit = closest.getPrimitive();
	float closestDistance = closest.getRay().length;

	float3 result = make_float3(Epsilon, Epsilon, Epsilon);
	if (closestDistance != FLT_MAX)
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

		float3 specColor = hitMat.getSpecularColor(uv.x, uv.y);
		bool hasSpec = specColor.x > 0 || specColor.y > 0 || specColor.z > 0;
		if (recursionDepth > 0 && hasSpec)
		{
			Vector3 d = closest.getRay().direction;
			Vector3 n = closest.getNormal();
			float rayNormalAngle = Vector3::Dot(d, n);
			//printf("n: (%f,%f,%f), d: (%f,%f,%f), a: %f \n", n.x, n.y, n.z, d.x, d.y, d.z, rayNormalAngle);
			//rayNormalAngle = rayNormalAngle < Epsilon ? Epsilon : rayNormalAngle > 1 ? 1 : rayNormalAngle;
			n *= (2 * rayNormalAngle);
			Vector3 newV = d - n;
			//Vector3 dir = newV.normalized();
			Ray bounced = Ray(closest.getRay().AsCoordinateVector() + (newV * Epsilon), newV);
			PrimaryTrace subTrace = PrimaryTrace(bounced);
			float3 subColor = subTrace.Do(scene, recursionDepth - 1);
			
			result.x += specColor.x * subColor.x;
			result.y += specColor.y * subColor.y;
			result.z += specColor.z * subColor.z;
		}

		float lightAttenuation = 1.f / (closestDistance * closestDistance);
		result.x *= lightAttenuation;
		result.y *= lightAttenuation;
		result.z *= lightAttenuation;
	}

	__syncthreads();
	return result;
}

__device__ float3 ShadowTrace::Do(Scene *scene, int recursionDepth)
{
	float3 lColor = light.getDiffuseColor();
	bool lit = !scene->shadowIntersect(ray);
	__syncthreads();

	float lightAttenuation = ((int)lit) / (ray.length * ray.length);
	float areaAttenuation = fmaxf(0.0f, Vector3::Dot(ri.getNormal(), ray.direction));

	lColor.x *= lightAttenuation * areaAttenuation;
	lColor.y *= lightAttenuation * areaAttenuation;
	lColor.z *= lightAttenuation * areaAttenuation;

	return lColor;
}

__device__ Ray ShadowTrace::createShadowRay(RayIntersection& ri, Light& l)
{
	Ray oldRay = ri.getRay();
	Vector3 oldRayEP = oldRay.AsCoordinateVector();
	Vector3 diff = l.getLocation() - oldRayEP;
	Vector3 dir = diff.normalized();
	Ray newRay = Ray(
		dir * Epsilon + oldRayEP,
		dir,
		diff.length() - Epsilon
	);
	return newRay;
}