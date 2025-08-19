#include <cuda_runtime.h>
#include <math_constants.h> // defines CUDART_INF
#include <cmath>
#include <algorithm>
#include "cuda/softmax.cuh"

template<typename S, typename D>
__global__ void softmaxKernelCUDA(
    const S* src_ptr, const shape_t src_shape, const strides_t src_strides,
    D* dst_ptr, const shape_t dst_shape, const strides_t dst_strides,
    uint8_t rank, uint8_t dim, size_t total_outer
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
 
    D max_val = -CUDART_INF;
    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        max_val = fmax(max_val, static_cast<D>(src_ptr[offset]));
    }
 
    D sum_exp = 0;
    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        sum_exp += exp(static_cast<D>(src_ptr[offset]) - max_val);
    }
 
    for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
        size_t offset = 0;
        for (int d = 0; d < rank; ++d) {
            size_t idx_val = (d == dim) ? i : cnt[d];
            offset += idx_val * src_strides.sizes[d];
        }
        dst_ptr[offset] = exp(static_cast<D>(src_ptr[offset]) - max_val) / sum_exp;
    }
}

template<typename S, typename D>
status launchSoftmaxCUDA(const tensor_t* src, tensor_t* dst, uint8_t dim, stream_t stream) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t total_outer = 1;
    for (int i = 0; i < src->rank; ++i)
        if (i != dim)
            total_outer *= src->shape.sizes[i];

    int threads = 256;
    int blocks = (total_outer + threads - 1) / threads;

    softmaxKernelCUDA<S, D><<<blocks, threads, 0, cudaStream>>>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<D*>(dst->address), dst->shape, dst->strides,
        src->rank, dim, total_outer
    );

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? SUCCESS : ERROR;
}

namespace cuda::nn {

status softmax(const tensor_t* src, tensor_t* dst, stream_t stream, uint8_t dim) {
    switch (src->dtype) {
        case float32: return launchSoftmaxCUDA<float, float>(src, dst, dim, stream);
        case float64: return launchSoftmaxCUDA<double, double>(src, dst, dim, stream);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cuda
