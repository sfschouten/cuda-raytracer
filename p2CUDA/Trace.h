#pragma once
#include <cuda_runtime.h>
#include <math.h>

#include "Scene.h"

class Trace
{
protected:
	const float Epsilon = 0.0005f;
	Ray ray;

public:
	__device__ Trace(Ray ray) : ray(ray) {}
	__device__ ~Trace() {}

	__device__ virtual float3 Do(Scene *scene, int recursionDepth) = 0;
};

class PrimaryTrace : public Trace
{
public:
	__device__ PrimaryTrace(Ray ray) : Trace(ray) {}
	__device__ ~PrimaryTrace() {}

	__device__ float3 Do(Scene *scene, int recursionDepth);
};


class ShadowTrace : public Trace
{
private:
	Light& light;
	RayIntersection& ri;

	__device__ Ray createShadowRay(RayIntersection& ri, Light& l);

public:
	__device__ ShadowTrace(RayIntersection& ri, Light& l)
		: Trace(createShadowRay(ri, l)), light(l), ri(ri) {}

	__device__ ~ShadowTrace() {}

	__device__ float3 Do(Scene *scene, int recursionDepth);
};

