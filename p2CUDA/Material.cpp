#include "Material.h"


Material::Material(float3 color)
	: color(color)
{}


__device__ __host__ Material::~Material()
{}


__device__ float3 Material::getColor(float u, float v)
{
	return color;
}