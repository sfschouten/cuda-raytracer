/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#ifndef _MANDELBROT_KERNEL_h_
#define _MANDELBROT_KERNEL_h_

#include <vector_types.h>
#include "Camera.h"
#include "Scene.h"

extern void RunRaytrace(uchar4 *dst, const int imageW, const int imageH, Camera *camera, Vector3 *directions, Scene *scene, bool cameraUnlocked);

#endif
