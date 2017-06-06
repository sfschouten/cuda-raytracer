#include "Plane.h"



Plane::Plane(Vector3 fromOrigin, Vector3 normal, Material material)
	: Primitive(material), normal(normal), fromOrigin(fromOrigin)
{
	d = -Vector3::Dot(normal, fromOrigin);
}

Plane::~Plane()
{}