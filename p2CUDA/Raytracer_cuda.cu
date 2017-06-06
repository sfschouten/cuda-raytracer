#include <stdio.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"
#include "Ray.h"
#include "Trace.h"
#include "Camera.h"
#include "Scene.h"

__global__ void Raytrace(uchar4 *dst, const int imageW, const int imageH, Camera camera, Vector3 *directions, Scene *scene, bool cameraUnlocked)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int i = y * imageW + x;

	Vector3 direction;
	if (cameraUnlocked) 
		direction = camera.getPixelDirection(x, y, imageW, imageH); 
	else 
		direction = directions[i]; 

	Ray ray = Ray(camera.getLocation(), direction);
	PrimaryTrace trace(ray);
	float3 color = trace.Do(scene, 8);

	dst[i].x = (color.x > 1 ? 1 : color.x < 0 ? 0 : color.x) * 255;
	dst[i].y = (color.y > 1 ? 1 : color.y < 0 ? 0 : color.y) * 255;
	dst[i].z = (color.z > 1 ? 1 : color.z < 0 ? 0 : color.z) * 255;
}

void RunRaytrace(uchar4 *dst, const int imageW, const int imageH, Camera camera, Vector3 *directions, Scene *scene, bool cameraUnlocked)
{
	dim3 block(8, 8, 1);
	dim3 grid(imageW / block.x, imageH / block.y, 1);
	Raytrace<<<grid, block>>>(dst, imageW, imageH, camera, directions, scene, cameraUnlocked);
} 
