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
	direction.Normalize();
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
	fp = Vector3::Cross(direction, a); fp.Normalize();
	sp = Vector3::Cross(fp, direction); sp.Normalize();
}

Vector3 Camera::getPixelDirection(int x, int y, int pWidth, int pHeight)
{
	float screenHeight = screenWidth * (pHeight / (float)pWidth);
	float rx = (float)x / pWidth * screenWidth - screenWidth / 2;
	float ry = (float)y / pHeight * -screenHeight + screenHeight / 2;
	Vector3 a = fp * -rx;
	Vector3 b = sp *  ry;
	Vector3 c = direction * screenDistance;
	a += b; a += c; a.Normalize();
	return a;
}