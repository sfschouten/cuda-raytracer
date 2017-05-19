#include "Vector3.h"


Vector3::Vector3(float x, float y, float z) {
	X = x;
	Y = y;
	Z = z;
};

Vector3::~Vector3()
{
}

Vector3 Vector3::operator+(const Vector3 other) {
	return *new Vector3(X + other.X, Y + other.Y, Z + other.Z);
}

Vector3 Vector3::operator*(const float scalar) {
	return *new Vector3(X * scalar, Y * scalar, Z * scalar);
}

