#include <math.h>

#include "Vector3.h"


__host__ __device__ Vector3 Vector3::Cross(Vector3 a, Vector3 b)
{
	Vector3 result(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); return result;
}
__host__ __device__ float Vector3::Dot(Vector3 a, Vector3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vector3::Vector3(float x, float y, float z)
	: x(x), y(y), z(z)
{}

__host__ __device__ Vector3::Vector3()
{}

__host__ __device__ Vector3 Vector3::operator+(const Vector3 other)
{
	Vector3 v(x + other.x, y + other.y, z + other.z); return v;
}
__host__ __device__ Vector3 Vector3::operator-(const Vector3 other)
{
	Vector3 v(x - other.x, y - other.y, z - other.z); return v;
}
__host__ __device__ Vector3 Vector3::operator*(const float scalar)
{
	Vector3 v(x * scalar, y * scalar, z * scalar); return v;
}
__host__ __device__ Vector3 Vector3::operator/(const float scalar)
{
	Vector3 v(x / scalar, y / scalar, z / scalar); return v;
}

__host__ __device__ Vector3 Vector3::operator+=(const Vector3 f)
{
	x += f.x; y += f.y; z += f.z; return *this;
}
__host__ __device__ Vector3 Vector3::operator-=(const Vector3 f)
{
	x -= f.x; y -= f.y; z -= f.z; return *this;
}

__host__ __device__ Vector3 Vector3::operator*=(const float f)
{
	x *= f; y *= f; z *= f; return *this;
}
__host__ __device__ Vector3 Vector3::operator/=(const float f)
{
	x /= f; y /= f; z /= f; return *this;
}

__host__ __device__ Vector3 Vector3::operator+=(const float f)
{
	x += f; y += f; z += f; return *this;
}
__host__ __device__ Vector3 Vector3::operator-=(const float f)
{
	x -= f; y -= f; z -= f; return *this;
}

__host__ __device__ Vector3 Vector3::normalized()
{
	return *this / length();
}
__host__ __device__ void Vector3::normalize()
{
	*this /= length();
}

__host__ __device__ float Vector3::length()
{
	return sqrt(x*x + y*y + z*z);
}