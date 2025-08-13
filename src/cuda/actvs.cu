#include <cmath>
#include <stdexcept>
#include <vector>
#include <array> 
#include "cuda/actvs.cuh" 

namespace {

template<typename S, typename D, class Actv>
__global__ void scalarActvKernel(const S* src, D* dst, Actv fn) { 
    *dst = fn(*src);
}  

template<typename S, typename D, class Actv>
__global__ void batchedActvKernel(
    const S* src, shape_t src_shape, strides_t src_strides,
    D* dst, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne, Actv fn
) { 
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) { 
        size_t offs = 0;
        size_t remaining = idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % dst_shape.sizes[dim];
            remaining /= dst_shape.sizes[dim];
 
            size_t src_idx = (src_shape.sizes[dim] == 1) ? 0 : dim_idx;
            offs += src_idx * src_strides.sizes[dim];
        }

        dst[idx] = fn(src[offs]);
    }
}   

template<typename S, typename D, class Actv, class ... Args>
status launchActvKernel(const tensor_t* src, tensor_t* dst, stream_t stream, Args... args)  { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    Actv fn(std::forward<Args>(args)...);
    if (src->rank == 0) {
        scalarActvKernel<S, D, Actv><<<1, 1, 0, cudaStream>>>(
            (const S*)(src->address),
            (D*)(dst->address), fn
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedActvKernel<S, D, Actv><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne, fn
        ); 
    }  
    return SUCCESS;
}  

struct ReLU {
    template <class T>
    __device__ __forceinline__ T operator()(T x) const noexcept { 
        return max(T(0), x);
    }
};

struct SiLU {
    template <class T>
    __device__ __forceinline__ T operator()(T x) const noexcept { 
        return x / (T(1) + exp(-x));
    }
};

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}   

constexpr static status launchDefaultKernel(const tensor_t* src, tensor_t* dst, stream_t) {
    return UNSUPPORTED_DTYPE;
};  
 
using Kernel = status(*)( const tensor_t*, tensor_t*, stream_t);      
 
constexpr auto dispatchReLUKernel = []() {
    std::array<Kernel, index(TYPES)> table; table.fill(launchDefaultKernel);
    table[index(int8)]    = launchActvKernel<int8_t, int8_t, ReLU>;
    table[index(int16)]   = launchActvKernel<int16_t, int16_t, ReLU>;
    table[index(int32)]   = launchActvKernel<int32_t, int32_t, ReLU>;
    table[index(int64)]   = launchActvKernel<int64_t, int64_t, ReLU>; 
    table[index(float32)] = launchActvKernel<float, float, ReLU>;
    table[index(float64)] = launchActvKernel<double, double, ReLU>;
    return table;
}();

constexpr auto dispatchSiLUKernel = []() { 
    std::array<Kernel, index(TYPES)> table; table.fill(launchDefaultKernel);
    table[index(float32)] = launchActvKernel<float, float, SiLU>;
    table[index(float64)] = launchActvKernel<double, double, SiLU>;
    return table;
}();
 
} namespace cuda {
 
status relu(const tensor_t* src, tensor_t* dst, stream_t stream) {    
    return dispatchReLUKernel[index(src->dtype)](src, dst, stream);
};

status silu(const tensor_t* src, tensor_t* dst, stream_t stream) { 
    return dispatchSiLUKernel[index(src->dtype)](src, dst, stream); 
};
  

}