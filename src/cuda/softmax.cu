#include <cuda_runtime.h>
#include <math_constants.h> // defines CUDART_INF
#include <cmath>
#include <cuda_fp16.h>
#include <algorithm>
#include "cuda/softmax.cuh" 

struct MaxOp {
    template<typename T> __device__ T identity() const { return -CUDART_INF; }
    template<typename T> __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

struct ExpSumOp { 
    template<typename T>
    __device__ __host__ T exp_fn(T x) const { return exp(x); }
 
    __device__ __host__ __half exp_fn(__half x) const {
        return __float2half(expf(__half2float(x)));
    }

    template<typename T>
    __device__ T identity() const { return T(0); }

    template<typename T>
    __device__ T operator()(T a, T b) const { return a + exp_fn(b); }
};

template<typename S, typename D>
__global__ void softmaxContiguousKernel(
    const S* src_ptr, size_t outer_size, size_t reduce_dim, size_t inner_size,
    D* dst_ptr
) {
    MaxOp maxop;
    ExpSumOp expsumop;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outer_size * inner_size) return;

    size_t outer = idx / inner_size;
    size_t inner = idx % inner_size;

    const S* slice_start = src_ptr + outer * reduce_dim * inner_size + inner;

    D max_val = maxop.template identity<D>();
    for (size_t i = 0; i < reduce_dim; ++i) {
        D val = static_cast<D>(slice_start[i * inner_size]);
        max_val = maxop(max_val, val);
    }

    D sum_exp = expsumop.template identity<D>();
    for (size_t i = 0; i < reduce_dim; ++i) {
        D val = static_cast<D>(slice_start[i * inner_size]);
        sum_exp += expsumop.exp_fn(val - max_val);
    }

    for (size_t i = 0; i < reduce_dim; ++i) {
        D val = static_cast<D>(slice_start[i * inner_size]);
        dst_ptr[outer * reduce_dim * inner_size + i * inner_size + inner] =
            expsumop.exp_fn(val - max_val) / sum_exp;
    }
}

template<typename S, typename D>
__global__ void softmaxStridedKernel(
    const S* src_ptr, const shape_t src_shape, const strides_t src_strides,
    D* dst_ptr, uint8_t rank, uint8_t dim, size_t total_outer
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outer) return;

    size_t cnt[8] = {0};
    size_t tmp = idx;
    for (int d = rank - 1; d >= 0; --d) {
        if (d == dim) continue;
        cnt[d] = tmp % src_shape.sizes[d];
        tmp /= src_shape.sizes[d];
    }

    MaxOp max_op;
    D max_val = max_op.template identity<D>();
    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        D val = static_cast<D>(src_ptr[offset]);
        max_val = max_op(max_val, val);
    }

    ExpSumOp sum_op;
    D sum_exp = sum_op.template identity<D>();
    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        D val = static_cast<D>(src_ptr[offset]);
        sum_exp += sum_op.exp_fn(val - max_val);
    }

    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        D val = static_cast<D>(src_ptr[offset]);
        dst_ptr[offset] = sum_op.exp_fn(val - max_val) / sum_exp;
    }
}

template<typename S, typename D>
status launchSoftmaxKernel(const tensor_t* src, tensor_t* dst, uint8_t dim, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);

    shape_t src_shape;
    strides_t src_strides;
    size_t total_outer = 1;
    for (int i = 0; i < src->rank; ++i) {
        src_shape.sizes[i] = src->shape.sizes[i];
        src_strides.sizes[i] = src->strides.sizes[i];
        if (i != dim) total_outer *= src_shape.sizes[i];
    }
 
    int threads = 256;
    int blocks = (total_outer + threads - 1) / threads;

    if (src->layout == CONTIGUOUS) {
        size_t outer_size = 1;
        size_t inner_size = 1;
        size_t reduce_dim = src_shape.sizes[dim];

        for (int i = 0; i < dim; ++i) outer_size *= src_shape.sizes[i];
        for (int i = dim + 1; i < src->rank; ++i) inner_size *= src_shape.sizes[i];

        softmaxContiguousKernel<S, D><<<blocks, threads, 0, cudaStream>>>(
            reinterpret_cast<const S*>(src->address), outer_size, reduce_dim, inner_size,
            reinterpret_cast<D*>(dst->address)
        );
    } 
    
    else {
        softmaxStridedKernel<S, D><<<blocks, threads, 0, cudaStream>>>(
            reinterpret_cast<const S*>(src->address), src_shape, src_strides,
            reinterpret_cast<D*>(dst->address), src->rank, dim, total_outer
        );
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SUCCESS : ERROR;
}


namespace cuda::nn {

status softmax(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case float16: return launchSoftmaxKernel<__half, __half>(src, dst, dim, stream);
        case float32: return launchSoftmaxKernel<float, float>(src, dst, dim, stream);
        case float64: return launchSoftmaxKernel<double, double>(src, dst, dim, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cuda
