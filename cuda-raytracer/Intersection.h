#pragma once
#include <math.h>
#include <math_constants.h>

#include "Ray.h"
#include "Primitive.h"
#include "Plane.h"
#include "Sphere.h"

class Intersection
{
protected:
	__device__ Intersection() {}
	__device__ ~Intersection() {}
	
public:
	__device__ virtual bool find() { return false; };
};

class RayIntersection : public Intersection
{

protected:
	Ray ray;
	Primitive *primitive;

	Vector3 collissionCoord;
	Vector3 normal;

public:
	__device__ RayIntersection(Ray ray, Primitive *primitive);
	__device__ ~RayIntersection() {}

	__device__ Ray &getRay() { return ray; }
	__device__ Vector3 &getNormal() { return normal; }
	__device__ Primitive *getPrimitive() { return primitive; }

	__device__ virtual float2 getTextureCoord();
};

class RaySphereIntersection : public RayIntersection
{
public:
	__device__ RaySphereIntersection(Ray ray, Sphere *s);

	__device__ bool find();
	__device__ float2 getTextureCoord();
};

class RayPlaneIntersection : public RayIntersection
{
public:
	__device__ RayPlaneIntersection(Ray ray, Plane *p);

	__device__ bool find();
	__device__ float2 getTextureCoord();
};