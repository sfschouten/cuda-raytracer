#pragma once
#include <cuda_runtime.h>

class Material
{
private:
	float3 color;

public:
	Material(float3 color);
	__device__ __host__ ~Material();

	__device__ float3 getColor(float u, float v);
};

