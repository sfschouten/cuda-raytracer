#pragma once
#include <vector>

#include "Primitive.h"
#include "Light.h"
#include "Ray.h"
#include "Intersection.h"
#include "ICudaObject.h"

class Scene
{
private:
	Primitive skyDome;
	
	std::vector<Light> h_Lights;
	std::vector<Sphere> h_Spheres;
	std::vector<Plane> h_Planes;

	Light *d_Lights;
	Sphere *d_Spheres;
	Plane *d_Planes;

	int nrSpheres;
	int nrPlanes;
	int nrLights;

public:
	Scene(Primitive skyDome);
	~Scene();
	
	void addLight(Light *l);
	void addSphere(Sphere *s);
	void addPlane(Plane *p);

	Scene *copyToDevice();

	__device__ int getNrLights() { return nrLights; }
	__device__ Light *getLights() { return d_Lights; }

	__device__ RayIntersection intersect(Ray ray);
	__device__ bool shadowIntersect(Ray ray);
};

