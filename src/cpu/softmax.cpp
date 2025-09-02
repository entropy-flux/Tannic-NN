#include <cmath>
#include <algorithm>
#include "cpu/softmax.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include "cpu/softmax.hpp"
#ifndef HAS_FLOAT16
    #if defined(__STDCPP_FLOAT16_T__) && __STDCPP_FLOAT16_T__
        #include <stdfloat>
        using half = std::float16_t;
        #define HAS_FLOAT16 1
    #else 
        #define HAS_FLOAT16 0 
        struct half_placeholder { float value; };
        using half = half_placeholder;
    #endif
#endif

namespace {
 
struct MaxOp {
    template<typename T>
    T identity() const { return -std::numeric_limits<T>::infinity(); }

    template<typename T>
    T operator()(T a, T b) const { return std::max(a, b); }
};

struct ExpSumOp {
    template<typename T>
    T exp_fn(T x) const { return std::exp(x); }

    template<typename T>
    T identity() const { return T(0); }

    template<typename T>
    T operator()(T a, T b) const { return a + exp_fn(b); }
};

template<typename S, typename D>
void softmaxContiguousKernel(
    const S* src_ptr, size_t outer_size, size_t reduce_dim, size_t inner_size,
    D* dst_ptr
) {
    MaxOp maxop;
    ExpSumOp expsumop;

    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            const S* slice_start = src_ptr + outer * reduce_dim * inner_size + inner;

            D max_val = maxop.template identity<D>();
            for (size_t i = 0; i < reduce_dim; ++i)
                max_val = maxop(max_val, static_cast<D>(slice_start[i * inner_size]));

            D sum_exp = expsumop.template identity<D>();
            for (size_t i = 0; i < reduce_dim; ++i)
                sum_exp += expsumop.exp_fn(static_cast<D>(slice_start[i * inner_size]) - max_val);

            for (size_t i = 0; i < reduce_dim; ++i)
                dst_ptr[outer * reduce_dim * inner_size + i * inner_size + inner] =
                    expsumop.exp_fn(static_cast<D>(slice_start[i * inner_size]) - max_val) / sum_exp;
        }
    }
} 


template<typename S, typename D>
void softmaxStridedKernel(
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,
    D* dst_ptr, uint8_t rank, uint8_t dim
) {
    size_t total_outer = 1;
    for (int i = 0; i < rank; ++i)
        if (i != dim) total_outer *= src_shape.sizes[i];

    size_t cnt[8] = {0};
    MaxOp maxop;
    ExpSumOp expsumop;

    for (size_t idx = 0; idx < total_outer; ++idx) { 
        size_t offset_base[8] = {0};
        size_t tmp = idx;
        for (int d = rank - 1; d >= 0; --d) {
            if (d == dim) continue;
            cnt[d] = tmp % src_shape.sizes[d];
            tmp /= src_shape.sizes[d];
            offset_base[d] = cnt[d] * src_strides.sizes[d];
        }

        D max_val = maxop.template identity<D>();
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = i * src_strides.sizes[dim];
            for (int d = 0; d < rank; ++d)
                if (d != dim) offset += offset_base[d];

            max_val = maxop(max_val, static_cast<D>(src_ptr[offset]));
        }

        D sum_exp = expsumop.template identity<D>();
        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = i * src_strides.sizes[dim];
            for (int d = 0; d < rank; ++d)
                if (d != dim) offset += offset_base[d];

            sum_exp += expsumop.exp_fn(static_cast<D>(src_ptr[offset]) - max_val);
        }

        for (size_t i = 0; i < src_shape.sizes[dim]; ++i) {
            size_t offset = i * src_strides.sizes[dim];
            for (int d = 0; d < rank; ++d)
                if (d != dim) offset += offset_base[d];

            dst_ptr[offset] = expsumop.exp_fn(static_cast<D>(src_ptr[offset]) - max_val) / sum_exp;
        }
    }
}
 
template<typename S, typename D>
status launchSoftmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    if (src->layout == CONTIGUOUS) { 
        size_t outer_size = 1, inner_size = 1;
        size_t reduce_dim = src->shape.sizes[dim];
        for (uint8_t i = 0; i < dim; ++i) outer_size *= src->shape.sizes[i];
        for (uint8_t i = dim + 1; i < src->rank; ++i) inner_size *= src->shape.sizes[i];

        softmaxContiguousKernel<S, D>(
            reinterpret_cast<const S*>(src->address),
            outer_size, reduce_dim, inner_size,
            reinterpret_cast<D*>(dst->address)
        );
    } 
    
    else {    
        shape_t src_shape;
        strides_t src_strides;
        for(uint8_t dim = 0; dim < src->rank; ++dim) {
            src_shape.sizes[dim] = src_strides.sizes[dim];
            src_strides.sizes[dim] = src_strides.sizes[dim];
        } 

        softmaxStridedKernel<S, D>(
            reinterpret_cast<const S*>(src->address),
            src->shape, src->strides,
            reinterpret_cast<D*>(dst->address),
            src->rank, dim
        );
    }
    return SUCCESS;
}

} // anonymous namespace

namespace cpu::nn {

status softmax(const tensor_t* src, tensor_t* dst, uint8_t dim) {
    switch (src->dtype) { 
#ifdef HAS_FLOAT16 
        case float16: return launchSoftmax<half, half> (src, dst, dim);
#endif
        case float32: return launchSoftmax<float, float>(src, dst, dim);
        case float64: return launchSoftmax<double, double>(src, dst, dim);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu
