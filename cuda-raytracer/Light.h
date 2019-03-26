#pragma once
#include "Vector3.h"

class Light
{
private:
	Vector3 location;
	float3 color;

public:
	Light(Vector3 location, float3 color) : location(location), color(color) {}

	Light(Vector3 location, float intensity, int r, int g, int b) 
		: color(make_float3(r * intensity / 255.f, g * intensity / 255.f, b * intensity / 255.f)),
		location(location) {}

	__host__ __device__ ~Light() {}

	__device__ Vector3& getLocation() { return location; }
	__device__ float3& getDiffuseColor() { return color; }
};

