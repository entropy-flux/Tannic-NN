#pragma once  
#include <tannic/runtime/resources.h>

using namespace tannic;

namespace cuda::nn { 

// TODO: Add backend api in base library to avoid code repetition here.

int getDeviceCount();
void setDevice(int /*device id*/);
void* allocate(const device_t*, size_t /*number of bytes*/);
void* deallocate(const device_t*, void*  /*memory address*/); 
void copyFromHost(const device_t*, const void* /*host memory address*/, void* /*device memory address*/, size_t /*number of bytes*/);  
bool compareFromHost(const device_t*, const void* /*host memory address*/, const void* /*device memory address*/, size_t /*number of bytes*/);

} // namespace cuda 