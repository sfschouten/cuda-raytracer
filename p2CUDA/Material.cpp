#include "Material.h"


Material::Material(float3 color)
	: color(color)
{}


Material::~Material()
{}


float3 Material::getColor(float2 uv)
{
	return color;
}