#pragma once
#include <cuda_runtime.h>

#include "Material.h"
#include "ICudaObject.h"

class Primitive
{
private:
	Material material;

public:
	Primitive(Material *material);
	__device__ __host__ ~Primitive();
};

