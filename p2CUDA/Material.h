#pragma once
#include <cuda_runtime.h>

class Material
{
private:
	float3 color;

public:
	Material(float3 color);
	__device__ __host__ ~Material();

	float3 getColor(float2 uv);
};

