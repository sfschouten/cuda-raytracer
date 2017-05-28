#include <stdio.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include "Ray.h"
#include "Trace.h"
#include "Camera.h"
#include "Scene.h"

__global__ void Raytrace(uchar4 *dst, const int imageW, const int imageH, Camera *camera, Vector3 *directions, Scene *scene)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int i = y * imageW + x;
	
	Vector3 direction = directions[i];
	Ray ray(camera->getLocation(), direction);
	PrimaryTrace trace(ray);
	float3 color = trace.Do(scene, 8);

	dst[i].x = color.x * 255;
	dst[i].y = color.y * 255;
	dst[i].z = color.z * 255;
	__syncthreads();
}

void RunRaytrace(uchar4 *dst, const int imageW, const int imageH, Camera *camera, Vector3 *directions, Scene *scene)
{
	//printf("%i %i", imageW, imageH);
	dim3 block(32, 32, 1);
	dim3 grid(imageW / block.x, imageH / block.y, 1);
	Raytrace<<<grid, block>>>(dst, imageW, imageH, camera, directions, scene);
} 
