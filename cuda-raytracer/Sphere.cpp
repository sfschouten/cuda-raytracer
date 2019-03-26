#include "Sphere.h"


Sphere::Sphere(float radius, Vector3 location, Material material)
	: Primitive(material), radius(radius), location(location)
{}

Sphere::~Sphere()
{}


__device__ __host__ Vector3& Sphere::getLocation()
{
	return location;
}

__device__ __host__ float Sphere::getRadius()
{
	return radius;
}
