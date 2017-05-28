#include "Material.h"


Material::Material(float3 diffuseColor, float3 specularColor)
	: diffuseColor(diffuseColor), specularColor(specularColor)
{}


__device__ __host__ Material::~Material()
{}


__device__ float3 Material::getDiffuseColor(float u, float v)
{
	return diffuseColor;
}
__device__ float3 Material::getSpecularColor(float u, float v)
{
	return specularColor;
}