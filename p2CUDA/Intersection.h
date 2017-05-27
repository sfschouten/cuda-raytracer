#pragma once
#include <math.h>
#include "Ray.h"
#include "Primitive.h"
#include "Plane.h"
#include "Sphere.h"

class Intersection
{
protected:
	__device__ Intersection();
	__device__ ~Intersection();
	
public:
	__device__ virtual void find() {};
};

class RayIntersection : public Intersection
{
protected:
	Ray ray;
	Primitive *p;
	Vector3 collissionCoord;
	Vector3 normal;

	__device__ RayIntersection(Ray ray, Primitive *p);

	virtual float2 getTextureCoord();

public:
	__device__ RayIntersection();
	__device__ ~RayIntersection();

	__device__ Ray getRay();
	__device__ Vector3 getNormal();
	//__device__ Ray getRay();
};

class RaySphereIntersection : public RayIntersection
{
public:
	__device__ RaySphereIntersection(Ray ray, Sphere *s);

	__device__ void find();
	float2 getTextureCoord();
};

class RayPlaneIntersection : public RayIntersection
{
public:
	__device__ RayPlaneIntersection(Ray ray, Plane *p);

	__device__ void find();
	float2 getTextureCoord();
};