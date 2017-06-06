#pragma once
#include "Primitive.h"
#include "Vector3.h"

class Sphere : public Primitive
{
private:
	Vector3 location;
	float radius;

public:
	Sphere(float radius, Vector3 location, Material material);
	__device__ __host__ ~Sphere();

	__device__ __host__ Vector3& getLocation();
	__device__ __host__ float getRadius();
};

