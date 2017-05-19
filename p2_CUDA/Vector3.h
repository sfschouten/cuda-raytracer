#pragma once
class Vector3
{
public:
	float X, Y, Z;

	Vector3(float x, float y, float z);
	~Vector3();

	Vector3 operator+(const Vector3 other);
	Vector3 operator*(const float scalar);

};

