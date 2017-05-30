#include "Material.h"


Material::Material(float3 diffuseColor, float3 specularColor)
	: diffuseColor(diffuseColor), specularColor(specularColor)
{}


__device__ __host__ Material::~Material()
{}


__device__ float3 Material::getDiffuseColor(float u, float v)
{
	if (test && ( (u < 0.5f && v < 0.5f) || (u > 0.5f && v > 0.5f) ))
	{
		return make_float3(0, 0, 0);
	}
	else
		return diffuseColor;
}
__device__ float3 Material::getSpecularColor(float u, float v)
{
	return specularColor;
}