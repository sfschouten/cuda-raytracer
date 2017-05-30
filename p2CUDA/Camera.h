#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "Vector3.h"

class Camera
{
private:
	const float screenWidth = 2;

	const double pi = 3.1415926535897;
	const double degToRad = pi / 180;

	Vector3 location, direction, target;
	int FOV;
	float screenDistance;

	Vector3 fp, sp;

	void setDirection();
	void setScreenDistance();

public:
	Camera(Vector3 location, Vector3 target, int FOV = 90);
	__device__ Camera(const Camera& c);
	__device__ __host__ ~Camera();

	void update();

	__device__ __host__ Vector3 getLocation();

	__device__ __host__ Vector3 getPixelDirection(int x, int y, int pWidth, int pHeight);

	void move(Vector3 diff);
	void moveDirection(float x, float y);

	void addFOV(float dFov) { FOV += dFov; setScreenDistance(); }
	
};

