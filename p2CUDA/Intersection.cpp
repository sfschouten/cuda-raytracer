#include "Intersection.h"

/*
*  Intersection
*/
__device__ Intersection::Intersection()
{}
__device__ Intersection::~Intersection()
{}

/*
*  RayIntersection
*/
__device__ RayIntersection::RayIntersection()
{}

__device__ RayIntersection::RayIntersection(Ray ray, Primitive *p)
	: ray(ray), p(p)
{}

__device__ RayIntersection::~RayIntersection()
{}

__device__ Ray RayIntersection::getRay()
{
	return ray;
}

__device__ Vector3 RayIntersection::getNormal()
{
	return normal;
}

float2 RayIntersection::getTextureCoord()
{ 
	return make_float2(0, 0); 
}


/*
*  RaySphereIntersection
*/
__device__ RaySphereIntersection::RaySphereIntersection(Ray ray, Sphere *s)
	: RayIntersection(ray, s)
{}

__device__ void RaySphereIntersection::find()
{
	Sphere *s = (Sphere *)p;
	Vector3 c = s->getLocation() - ray.origin;
	float t = Vector3::Dot(c, ray.direction);
	Vector3 q = c - ray.direction * t;
	float p2 = Vector3::Dot(q, q);
	float sphere_r2 = s->getRadius() * s->getRadius();
	if (p2 > sphere_r2)
		return;

	t -= (float)sqrt(sphere_r2 - p2);
	if (t < ray.length && t > 0)
	{
		ray.length = t;
		collissionCoord = ray.AsCoordinateVector();
		normal = (collissionCoord - s->getLocation()).Normalized();
	}
}

float2 RaySphereIntersection::getTextureCoord()
{
	return make_float2(0, 0);
}

/*
*  RayPlaneIntersection
*/
__device__ RayPlaneIntersection::RayPlaneIntersection(Ray ray, Plane *p)
	: RayIntersection(ray, p)
{}

__device__ void RayPlaneIntersection::find()
{}

float2 RayPlaneIntersection::getTextureCoord()
{
	return make_float2(0, 0);
}