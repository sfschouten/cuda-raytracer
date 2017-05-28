#pragma once
#include "Primitive.h"
#include "Vector3.h"

class Plane : public Primitive
{
private:
	Vector3 fromOrigin;
	Vector3 normal;

public:
	Plane(Vector3 fromOrigin, Vector3 normal, Material material);
	__device__ __host__ ~Plane();

};

