#pragma once
#include <cfloat>
#include "Vector3.h"

struct Ray
{
	Vector3 origin;
	Vector3 direction;
	float length = FLT_MAX;

	__host__ __device__ Ray();
	__host__ __device__ Ray(Vector3 origin, Vector3 direction);

	__host__ __device__ Vector3 AsCoordinateVector();
};