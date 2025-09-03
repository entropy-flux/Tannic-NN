#include <cmath>
#include <stdexcept>
#include <vector>
#include <array> 
#include <cuda_fp16.h>
#include "cuda/actvs.cuh" 

namespace {
 
template<typename S, typename D, class Op>
__global__ void singletonActvKernel(const S* __restrict__ src, D* __restrict__ dst, Op op) { 
    *dst = op(*src);
}  

template<typename S, typename D, class Op>
__global__ void contiguousActvKernel(const S* __restrict__ src, D* __restrict__ dst, size_t ne, Op op) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) {
        dst[idx] = op(src[idx]);
    }
}
 
template<typename S, typename D, class Op>
__global__ void stridedActvKernel(
    const S* __restrict__ src_ptr, strides_t src_strides,    
    D* __restrict__ dst_ptr, shape_t resets,          
    uint8_t dst_rank, size_t ne, Op op
){
    int rank = static_cast<int>(dst_rank);
    const size_t gstride = size_t(blockDim.x) * gridDim.x;
    for (size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < ne; idx += gstride) {
        size_t offset = 0;
        size_t divisor = 1;

        for (int dim = rank - 1; dim >= 0; --dim) { 
            const size_t extent    = resets.sizes[dim] / src_strides.sizes[dim];
            const size_t coord     = (idx / divisor) % extent; 
            offset += coord * src_strides.sizes[dim];
            divisor *= extent;
        }

        dst_ptr[idx] = op(src_ptr[offset]);
    }
} 
 
template<typename S, typename D, class Op, class ... Args>
status launchActvKernel(const tensor_t* src, tensor_t* dst, stream_t stream, Args... args)  { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    Op op(std::forward<Args>(args)...);

    size_t ne = dst->size; 
    size_t blockSize = 256;
    size_t gridSize = (ne + blockSize - 1) / blockSize; 

    if (src->layout == SINGLETON) {
        singletonActvKernel<S, D, Op><<<1, 1, 0, cudaStream>>>(
            (const S*)(src->address),
            (D*)(dst->address),
            op
        ); 
        return SUCCESS;
    }

    else if (src->layout == CONTIGUOUS) {
        contiguousActvKernel<S, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address),
            (D*)(dst->address),
            ne,
            op
        );
        return SUCCESS;
    }
    
    else {
        strides_t strides{0};
        shape_t resets{0};
        for (int dim = 0; dim < src->rank; ++dim) {
            resets.sizes[dim] = dst->shape.sizes[dim] * src->strides.sizes[dim];
            strides.sizes[dim] = src->strides.sizes[dim];
        } 
        
        stridedActvKernel<S, D, Op><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address), strides,
            (D*)(dst->address), resets,
            src->rank, ne,
            op
        );
        return SUCCESS;
    }
}    

struct ReLU {
    template <class T>
    __device__ __forceinline__ T operator()(T x) const noexcept { 
        return max(T(0), x);
    } 

    __device__ __forceinline__ __half operator()(__half x) const noexcept {
        return __hgt(x, __float2half(0.0f)) ? x : __float2half(0.0f);
    }

};

struct SiLU {
    template <class T>
    __device__ __forceinline__ T operator()(T x) const noexcept { 
        return x / (T(1) + exp(-x));
    }

    __device__ __forceinline__ __half operator()(__half x) const noexcept {
        float x_float = __half2float(x);
        float result = x_float / (1.0f + expf(-x_float));
        return __float2half(result);
    }
};

struct GELU {
    template <class T>
    __device__ __forceinline__ T operator()(T x) const noexcept {
        const T c = rsqrt(T(M_PI) / T(2.0));  
        return T(0.5) * x * (T(1) + tanh(c * (x + T(0.044715) * x * x * x)));
    }

    __device__ __forceinline__ __half operator()(__half x) const noexcept {
        float x_float = __half2float(x);
        const float c = rsqrtf(float(M_PI) / 2.0f); 
        float result = 0.5f * x_float * (1.0f + tanhf(c * (x_float + 0.044715f * x_float * x_float * x_float)));
        return __float2half(result);
    }
 
};
  
} namespace cuda::nn {
    
status relu(const tensor_t* src, tensor_t* dst, stream_t stream) {    
    switch (src->dtype) {
        case int8:    return launchActvKernel<int8_t, int8_t, ReLU>(src, dst, stream);
        case int16:   return launchActvKernel<int16_t, int16_t, ReLU>(src, dst, stream);
        case int32:   return launchActvKernel<int32_t, int32_t, ReLU>(src, dst, stream);
        case int64:   return launchActvKernel<int64_t, int64_t, ReLU>(src, dst, stream);
        case float16: return launchActvKernel<__half, __half, ReLU>(src, dst, stream);
        case float32: return launchActvKernel<float, float, ReLU>(src, dst, stream);
        case float64: return launchActvKernel<double, double, ReLU>(src, dst, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
};

status silu(const tensor_t* src, tensor_t* dst, stream_t stream) { 
    switch (src->dtype) {
        case float16: return launchActvKernel<__half, __half, SiLU>(src, dst, stream);
        case float32: return launchActvKernel<float, float, SiLU>(src, dst, stream);
        case float64: return launchActvKernel<double, double, SiLU>(src, dst, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
};

status gelu(const tensor_t* src, tensor_t* dst, stream_t stream) { 
    switch (src->dtype) {
        case float16: return launchActvKernel<__half, __half, GELU>(src, dst, stream);
        case float32: return launchActvKernel<float, float, GELU>(src, dst, stream);
        case float64: return launchActvKernel<double, double, GELU>(src, dst, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
};

}