#include "Ray.h"
	

Ray::Ray()
{
}

Ray::~Ray()
{
}

Vector3 ToCoordinateVector() {
	return origin + direction * length;
}