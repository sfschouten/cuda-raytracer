#pragma once
#include <vector>
#include <typeinfo>

#include "Primitive.h"
#include "Ray.h"
#include "Intersection.h"
#include "ICudaObject.h"

class Scene
{
private:
	std::vector<Sphere> h_Spheres;
	std::vector<Plane> h_Planes;

	Sphere *d_Spheres;
	Plane *d_Planes;
	int nrSpheres;
	int nrPlanes;

public:
	Scene();
	~Scene();
	
	void addSphere(Sphere *s);
	void addPlane(Plane *p);
	
	Scene *copyToDevice();

	__device__ RayIntersection intersect(Ray ray);
};

