#pragma once
#include <cuda_runtime.h>


class ICudaObject
{
private:
	//ICudaObject *deviceObject;

public:
	//virtual ~ICudaObject() { if (deviceObject) cudaFree(deviceObject); }

	//virtual ICudaObject *copyToDevice() = 0;

//protected:
	template <class T>
	static T *physicalCopyToDevice(T *object)
	{
		T *d_Object;

		size_t size = sizeof(*object);
		cudaMalloc(&d_Object, size);
		cudaMemcpy(d_Object, object, size, cudaMemcpyHostToDevice);
		return d_Object;
	}
};