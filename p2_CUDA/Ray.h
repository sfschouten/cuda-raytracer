#pragma once
class Ray
{
public:
	Vector3 origin;
	Vector3 direction;
	float length;
	Ray();
	~Ray();
	Vector3 ToCoordinateVector();
	
};

