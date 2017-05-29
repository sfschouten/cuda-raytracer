#include "Scene.h"


Scene::Scene(Primitive skyDome)
	: skyDome(skyDome)
{
	h_Lights = std::vector<Light>();
	h_Spheres = std::vector<Sphere>();
	h_Planes = std::vector<Plane>();
}

Scene::~Scene()
{}

Scene *Scene::copyToDevice()
{
	int arraySize;
	
	nrLights = h_Lights.size();
	arraySize = sizeof(Light) * nrLights;
	if (arraySize)
	{
		cudaMalloc(&d_Lights, arraySize);
		cudaMemcpy(d_Lights, &h_Lights[0], arraySize, cudaMemcpyHostToDevice);
	}

	nrSpheres = h_Spheres.size();
	arraySize = sizeof(Sphere) * nrSpheres;
	if (arraySize)
	{
		cudaMalloc(&d_Spheres, arraySize);
		cudaMemcpy(d_Spheres, &h_Spheres[0], arraySize, cudaMemcpyHostToDevice);
	}
	
	nrPlanes = h_Planes.size();
	arraySize = sizeof(Plane) * nrPlanes;
	if (arraySize)
	{
		cudaMalloc(&d_Planes, arraySize);
		cudaMemcpy(d_Planes, &h_Planes[0], arraySize, cudaMemcpyHostToDevice);
	}

	return ICudaObject::physicalCopyToDevice<Scene>(this);
}

void Scene::addLight(Light *l)
{
	h_Lights.insert(h_Lights.end(), *l);
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
	RayIntersection closest = RayIntersection(ray, &skyDome);

	for (int i = 0; i < nrSpheres; i++)
	{
		Sphere *s = &d_Spheres[i];
		
		RaySphereIntersection isct = RaySphereIntersection(ray, s);
		
		if (isct.find() && closest.getRay().length > isct.getRay().length)
			closest = isct; 
	}

	for (int i = 0; i < nrPlanes; i++)
	{
		Plane *p = &d_Planes[i];

		RayPlaneIntersection isct = RayPlaneIntersection(ray, p);
		
		if (isct.find() && closest.getRay().length > isct.getRay().length)
		    closest = isct;
	}

	return closest;
}

__device__ bool Scene::shadowIntersect(Ray ray)
{
	bool inShadow = false;

	for (int i = 0; i < nrSpheres && !inShadow; i++)
	{
		Sphere *s = &d_Spheres[i];

		RaySphereIntersection isct = RaySphereIntersection(ray, s);
		if (isct.find())
			inShadow = true;
	}

	for (int i = 0; i < nrPlanes && !inShadow; i++)
	{
		Plane *p = &d_Planes[i];

		RayPlaneIntersection isct = RayPlaneIntersection(ray, p);
		if (isct.find())
			inShadow = true;
	}
	
	__syncthreads();
	return inShadow;
}