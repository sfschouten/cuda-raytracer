#include <stdio.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include "Ray.h"
#include "Camera.h"
#include "Scene.h"

__global__ void Raytrace(uchar4 *dst, const int imageW, const int imageH, Camera *camera, Vector3 *directions, Scene *scene)
{
	const float Epsilon = 0.00001f;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int i = y * imageW + x;
	
	Vector3 direction = directions[i];
	Ray ray(camera->getLocation(), direction);

	/*if (x % 64 == 0 && y % 64 == 0)
		printf("%i, %i: d=(%f, %f, %f) \n", x, y, direction.x, direction.y, direction.z);*/

	/*dst[i].x = 255 * direction.x;
	dst[i].y = 255 * direction.y;
	dst[i].z = 255 * direction.z;*/
	
	RayIntersection closest = scene->intersect(ray);
	Vector3 normal = closest.getNormal();

	//printf("%i, %i: n=(%f, %f, %f) l=%f \n", x, y, normal.x, normal.y, normal.z, closest->getRay().length);

	/*dst[i].x = 255 * normal.x;
	dst[i].y = 255 * normal.y;
	dst[i].z = 255 * normal.z;*/
	
	float distance = closest.getRay().length;
	dst[i].x = (distance / 8) * 255;
	dst[i].y = (distance / 8) * 255;
	dst[i].z = (distance / 8) * 255;
}

void RunRaytrace(uchar4 *dst, const int imageW, const int imageH, Camera *camera, Vector3 *directions, Scene *scene)
{
	dim3 block(16, 16, 1);
	dim3 grid(imageW / block.x, imageH / block.y, 1);
	Raytrace<<<grid, block>>>(dst, imageW, imageH, camera, directions, scene);
} 
