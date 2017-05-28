#pragma once
#include "Vector3.h"

class Light
{
private:
	Vector3 location;
	float3 color;

public:
	Light(Vector3 location, float3 color) : location(location), color(color) {}
	__host__ __device__ ~Light() {}

	__device__ Vector3 getLocation() { return location; }
	__device__ float3 getColor() { return color; }
};

