#include "Scene.h"


Scene::Scene()
{
	h_Spheres = std::vector<Sphere>();
	h_Planes = std::vector<Plane>();
}

Scene::~Scene()
{}

Scene *Scene::copyToDevice()
{
	int arraySize;
	
	nrSpheres = h_Spheres.size();
	arraySize = sizeof(Sphere) * h_Spheres.size();
	if (arraySize)
	{
		cudaMalloc(&d_Spheres, arraySize);
		cudaMemcpy(d_Spheres, &h_Spheres[0], arraySize, cudaMemcpyHostToDevice);
	}
	
	nrPlanes = h_Planes.size();
	arraySize = sizeof(Plane) * h_Planes.size();
	if (arraySize)
	{
		cudaMalloc(&d_Planes, arraySize);
		cudaMemcpy(d_Planes, &h_Planes[0], arraySize, cudaMemcpyHostToDevice);
	}

	return ICudaObject::physicalCopyToDevice<Scene>(this);
}

void Scene::addSphere(Sphere *s)
{
	h_Spheres.insert(h_Spheres.end(), *s);
}

void Scene::addPlane(Plane *p)
{
	h_Planes.insert(h_Planes.end(), *p);
}

__device__ RayIntersection Scene::intersect(Ray ray)
{
	RayIntersection closest;

	for (int i = 0; i < nrSpheres; i++)
	{
		Sphere *s = &d_Spheres[i];
		
		RaySphereIntersection isct = RaySphereIntersection(ray, s);
		isct.find();
		
		if (closest.getRay().length > isct.getRay().length) 
			closest = isct; 
	}

	for (int i = 0; i < nrPlanes; i++)
	{
		Plane *p = &d_Planes[i];

		RayPlaneIntersection isct = RayPlaneIntersection(ray, p);
		isct.find();

		if (closest.getRay().length > isct.getRay().length)
		    closest = isct;
	}

	return closest;
}