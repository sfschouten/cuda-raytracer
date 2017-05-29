#pragma once
#include "Primitive.h"
#include "Vector3.h"

class Plane : public Primitive
{
private:
	Vector3 fromOrigin;
	Vector3 normal;
	float d;

public:
	Plane(Vector3 fromOrigin, Vector3 normal, Material material);
	__device__ __host__ ~Plane();

	__device__ Vector3 getNormal() { return normal; }
	__device__ Vector3 getDistanceFromOrigin() { return fromOrigin; }
	__device__ float getD() { return d; }
};

