#pragma once
#include <cuda_runtime.h>

struct Vector3
{
public:
	float x, y, z;

	__host__ __device__ static Vector3 Cross(Vector3 a, Vector3 b);
	__host__ __device__ static float Dot(Vector3 a, Vector3 b);

	__host__ __device__ Vector3(float x, float y, float z);
	__host__ __device__ Vector3();

	__host__ __device__ Vector3 operator+(const Vector3 other);
	__host__ __device__ Vector3 operator-(const Vector3 other);
	__host__ __device__ Vector3 operator*(const float scalar);
	__host__ __device__ Vector3 operator/(const float scalar);

	__host__ __device__ Vector3 operator+=(const Vector3 f);
	__host__ __device__ Vector3 operator-=(const Vector3 f);

	__host__ __device__ Vector3 operator*=(const float f);
	__host__ __device__ Vector3 operator/=(const float f);
	__host__ __device__ Vector3 operator+=(const float f);
	__host__ __device__ Vector3 operator-=(const float f);

	__host__ __device__ Vector3 Normalized();
	__host__ __device__ void Normalize();
};

