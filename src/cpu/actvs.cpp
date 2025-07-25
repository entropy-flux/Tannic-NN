#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>
#include "cpu.hpp"   
 
template<typename S, typename D, class A>
void actvKernel(
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    const S* src = static_cast<const S*>(src_ptr);
    D* dst = static_cast<D*>(dst_ptr);
    A actv;

    if (rank == 0) {
        *dst = actv(*src);
        return;
    }

    size_t total = 1;
    for (uint8_t i = 0; i < rank; ++i)
        total *= dst_sz[i];

    for (size_t idx = 0; idx < total; ++idx) {
        size_t offs = 0;
        for (uint8_t dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_ne[dim];
        }

        dst[idx] = actv(src[offs]);

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_sz[dim])
                break;
            cnt[dim] = 0;
        }
    }
} 

void defaultKernel(
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    throw std::runtime_error("Not supported dtype");
};  

struct ReLU { 
    template<class A>
    auto operator()(A&& a) const noexcept {
        return std::max<A>(0, a);
    }
};

struct SiLU {
    template<class A>
    auto operator()(A&& x) const noexcept {
        return x / (1 + std::exp(-x));
    } 
};

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}  
 
using Kernel = void(*)( 
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
);

namespace actv {
  
constexpr auto relu = []() {
    std::array<Kernel, index(TYPES)> table; table.fill(defaultKernel);
    table[index(int8)] = actvKernel<int8_t, int8_t, ReLU>;
    table[index(int16)] = actvKernel<int16_t, int16_t, ReLU>;
    table[index(int32)] = actvKernel<int32_t, int32_t, ReLU>;
    table[index(int64)] = actvKernel<int64_t, int64_t, ReLU>; 
    table[index(float32)] = actvKernel<float, float, ReLU>;
    table[index(float64)] = actvKernel<double, double, ReLU>;
    return table;
}();

constexpr auto silu = []() { 
    std::array<Kernel, index(TYPES)> table; table.fill(defaultKernel);
    table[index(float32)] = actvKernel<float, float, SiLU>;
    table[index(float64)] = actvKernel<double, double, SiLU>;
    return table;
}();

} namespace cpu {
 
void relu(const tensor_t* src, tensor_t* dst) { 
    std::vector<size_t> cnt(src->rank, 0);
    actv::relu[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
};

void silu(const tensor_t* src, tensor_t* dst) { 
    std::vector<size_t> cnt(src->rank, 0);
    actv::silu[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
};
  
} // namespace cpu 