#include "Ray.h"


__host__ __device__ Ray::Ray(Vector3 origin, Vector3 direction, float length)
	: origin(origin), direction(direction), length(length)
{}

__host__ __device__ Vector3 Ray::AsCoordinateVector()
{
	return origin + direction * length;
}