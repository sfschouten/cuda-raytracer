#pragma once
#include "Camera.h"

__device__ Camera::Camera(const Camera& c)
  : location(c.location),
	direction(c.direction),
	target(c.target),
	FOV(c.FOV),
	screenDistance(c.screenDistance),
	fp(c.fp),
	sp(c.sp)
{}

Camera::Camera(Vector3 location, Vector3 target, int FOV)
{
	this->location = location;
	this->target = target;
	this->FOV = FOV;
	setDirection();
	setScreenDistance();
}

Camera::~Camera()
{}

void Camera::setDirection()
{
	direction = target - location; 
	direction.normalize();
}

void Camera::setScreenDistance()
{
	screenDistance = screenWidth / (2 * tan(FOV / 2 * degToRad));
}

Vector3 Camera::getLocation()
{
	return location;
}

void Camera::update()
{
	Vector3 a(direction.x, 1, direction.z);
	fp = Vector3::Cross(direction, a); fp.normalize();
	sp = Vector3::Cross(fp, direction); sp.normalize();
}

__device__ __host__ Vector3 Camera::getPixelDirection(int x, int y, int pWidth, int pHeight)
{
	float screenHeight = screenWidth * (pHeight / (float)pWidth);
	float rx = (float)x / pWidth * screenWidth - screenWidth / 2;
	float ry = (float)y / pHeight * -1 * screenHeight + screenHeight / 2;
	Vector3 a = fp * -rx;
	Vector3 b = sp * ry;
	Vector3 c = direction * screenDistance;
	a += b; a += c; a.normalize();
	return a;
}

void Camera::move(Vector3 diff)
{
	Vector3 a = fp * -diff.x;
	Vector3 b = sp * diff.y;
	Vector3 c = direction * diff.z;
	a += b; a += c;
	location += a;
}

void Camera::moveDirection(float x, float y)
{
	Vector3 a = fp * -x;
	Vector3 b = sp * y;
	Vector3 c = direction * screenDistance;
	a += b; a += c; a.normalize();
	direction = a;
	update();
}