#include <cmath>
#include <algorithm>
#include "cpu/softmax.hpp"

namespace {

template<typename S, typename D>
void softmaxKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, uint8_t dim
) {
    if (rank == 0) {
        *dst_ptr = 1; 
        return;
    }

    size_t total = 1;
    for (int i = 0; i < rank; ++i)
        if (i != dim)
            total *= src_shape.sizes[i];

    size_t cnt[8] = {0};

    for (size_t idx = 0; idx < total; ++idx) { 
        D max_val = -std::numeric_limits<D>::infinity();
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }
            max_val = std::max(max_val, static_cast<D>(src_ptr[offset]));
        }
 
        D sum_exp = D(0);
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }
            sum_exp += std::exp(static_cast<D>(src_ptr[offset]) - max_val);
        }
 
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = 0;
            for (int d = 0; d < rank; ++d) {
                size_t idx_val = (d == dim) ? i : cnt[d];
                offset += idx_val * src_strides.sizes[d];
            }
            dst_ptr[offset] = std::exp(static_cast<D>(src_ptr[offset]) - max_val) / sum_exp;
        }
 
        for (int d = rank - 1; d >= 0; --d) {
            if (d == dim) continue;
            if (++cnt[d] < src_shape.sizes[d]) break;
            cnt[d] = 0;
        }
    }
}

template<typename S, typename D>
status launchSoftmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    softmaxKernel<S, D>(
        reinterpret_cast<const S*>(src->address), src->shape, src->strides,
        reinterpret_cast<D*>(dst->address), dst->shape, dst->strides,
        src->rank, dim
    );
    return SUCCESS;
}

} // anonymous namespace

namespace cpu::nn {

status softmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) {
        case float32: return launchSoftmax<float, float>(src, dst, dim);
        case float64: return launchSoftmax<double, double>(src, dst, dim);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu
