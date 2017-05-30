#pragma once
#include <cuda_runtime.h>

class Material
{
private:
	float3 diffuseColor;
	float3 specularColor;

public:
	bool test = false;

	Material(float3 diffuseColor, float3 specularColor);
	__device__ __host__ ~Material();

	__device__ float3 getDiffuseColor(float u, float v);
	__device__ float3 getSpecularColor(float u, float v);
};

