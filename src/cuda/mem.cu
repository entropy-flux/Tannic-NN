#include <tannic/runtime/streams.h>
#include "cuda/exc.cuh"
#include "cuda/mem.cuh" 

namespace cuda::nn { 

// TODO: Add backend api in base library to avoid code repetition here.
  
int getDeviceCount() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count); CUDA_CHECK(err);
    return count;
}

void setDevice(int id) {
    CUDA_CHECK(cudaSetDevice(id));
}

void* allocate(const device_t* resource, size_t nbytes) {
    setDevice(resource->id); 
    void* ptr = nullptr;
    if (resource->traits & SYNC) { 
        CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    } else {
        stream_t stream = pop_stream(resource); 
        cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
        CUDA_CHECK(cudaMallocAsync(&ptr, nbytes, cudaStream));
        put_stream(resource, stream); 
    }
    return ptr;
} 

void* deallocate(const device_t* resource, void* ptr) {
    setDevice(resource->id);
    if (resource->traits & SYNC) {
        CUDA_CHECK(cudaFree(ptr));
    } else {
        stream_t stream = pop_stream(resource); 
        cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
        CUDA_CHECK(cudaFreeAsync(ptr, cudaStream));
        put_stream(resource, stream); 
    }
    return nullptr;
}

void copyFromHost(const device_t* resource, const void* src , void* dst, size_t nbytes) {
    setDevice(resource->id);
    if (resource->traits & SYNC) {
        cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    } 
    else {
        stream_t stream = pop_stream(resource); 
        cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, cudaStream);
        put_stream(resource, stream); 
    }
} 

bool compareFromHost(const device_t* resource, const void* hst_ptr, const void* dvc_ptr, size_t nbytes) {  
    void* buffer = malloc(nbytes); 
    CUDA_CHECK(cudaMemcpy(buffer, dvc_ptr, nbytes, cudaMemcpyDeviceToHost));
    bool result = (memcmp(hst_ptr, buffer, nbytes) == 0);
    free(buffer);   
    return result;
}

} // namespace cuda  