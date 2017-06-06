#include "Intersection.h"

/*
*  RayIntersection
*/

__device__ RayIntersection::RayIntersection(Ray ray, Primitive *primitive)
	: ray(ray), primitive(primitive)
{}

__device__ float2 RayIntersection::getTextureCoord()
{ 
	return make_float2(0, 0); 
}


/*
*  RaySphereIntersection
*/
__device__ RaySphereIntersection::RaySphereIntersection(Ray ray, Sphere *s)
	: RayIntersection(ray, s)
{}

__device__ bool RaySphereIntersection::find()
{
	bool result = false;

	Sphere *s = (Sphere *)primitive;
	Vector3 c = s->getLocation() - ray.origin;
	float t = Vector3::Dot(c, ray.direction);
	Vector3 q = c - ray.direction * t;
	float p2 = Vector3::Dot(q, q);
	float sphere_r2 = s->getRadius() * s->getRadius();
	if (p2 <= sphere_r2)
	{
		t -= (float)sqrt(sphere_r2 - p2);
		if (t <= ray.length && t > 0)
		{
			ray.length = t;
			collissionCoord = ray.AsCoordinateVector();
			normal = (collissionCoord - s->getLocation()).normalized();
			result = true;
		}
	}
	__syncthreads();
	return result;
}

__device__ float2 RaySphereIntersection::getTextureCoord()
{
	Sphere *s = (Sphere *)primitive;
	Vector3 hitVec = (collissionCoord - s->getLocation()) / s->getRadius();
	float u = 0.5f + (float)atan2(hitVec.z, hitVec.x) / (2 * CUDART_PI_F);
	float v = 0.5f - (float)asin(hitVec.y) / CUDART_PI_F;

	return make_float2(u, v);
}

/*
*  RayPlaneIntersection
*/
__device__ RayPlaneIntersection::RayPlaneIntersection(Ray ray, Plane *p)
	: RayIntersection(ray, p)
{
	normal = p->getNormal();
}

__device__ bool RayPlaneIntersection::find()
{
	bool result = false;

	Plane *plane = (Plane *)primitive;
	float a = Vector3::Dot(ray.direction, normal);
	if (a != 0)
	{
		float t = -1 * (Vector3::Dot(ray.origin, normal) + plane->getD()) / a;
		if (t <= ray.length && t > 0)
		{
			ray.length = t;
			collissionCoord = ray.AsCoordinateVector();
			result = true;
		}
	}

	__syncthreads();
	return result;
}

__device__ float2 RayPlaneIntersection::getTextureCoord()
{
	Plane *plane = (Plane *)primitive;
	Vector3 diff = collissionCoord - plane->getDistanceFromOrigin();

	float x = fmodf(diff.x, 1);
	if (x < 0)
		x += 1;

	float y = fmodf(diff.z, 1);
	if (y < 0)
		y += 1;

	return make_float2(x, y);
}