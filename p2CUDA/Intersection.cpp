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
	if (sphere_r2 >= p2)
	{
		t -= (float)sqrt(sphere_r2 - p2);
		if (t < ray.length && t > 0)
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
	return make_float2(0, 0);
}

/*
*  RayPlaneIntersection
*/
__device__ RayPlaneIntersection::RayPlaneIntersection(Ray ray, Plane *p)
	: RayIntersection(ray, p)
{}

__device__ bool RayPlaneIntersection::find()
{
	return false;
}

__device__ float2 RayPlaneIntersection::getTextureCoord()
{
	return make_float2(0, 0);
}