#include <cmath>
#include <stdexcept>
#include <vector>
#include <array> 
#include "cpu/actvs.hpp" 

namespace {

template<typename S, typename D, class Actv>
void scalarActvKernel(
    const S* src_ptr, D* dst_ptr, Actv fn
) {
    *dst_ptr = fn(*src_ptr);
}    

template<typename S, typename D, class Actv>
void batchedActvKernel( 
    const S* src_ptr, const shape_t& src_shape, const strides_t& src_strides,           
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides, 
    uint8_t rank, size_t ne, Actv fn
) {  
    size_t cnt[8] = {0};
    for (size_t idx = 0; idx < ne; ++idx) {
        size_t offs = 0;
        for (int dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_strides.sizes[dim];
        }

        dst_ptr[idx] = fn(src_ptr[offs]);

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_shape.sizes[dim])
                break;
            cnt[dim] = 0;
        }
    } 
}

template<typename S, typename D, class Actv, class ... Args>
status launchActvKernel(const tensor_t* src, tensor_t* dst, Args... args) {
    Actv fn(std::forward<Args>(args)...);

    if (src->rank == 0) {
        scalarActvKernel<S, D, Actv>(
            (const S*)(src->address), 
            (D*)(dst->address), fn
        ); 
    } 
    
    else {    
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        batchedActvKernel<S, D, Actv>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne, fn
        ); 
    } 
    return SUCCESS;
}        
  
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

struct GELU {
    template<class A>
    auto operator()(A&& x) const noexcept {
        const double c = std::sqrt(2.0 / M_PI);
        return static_cast<A>(0.5 * x * (1.0 + std::tanh(c * (x + 0.044715 * x * x * x))));
    }
};

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}   

constexpr static status launchDefaultKernel(const tensor_t* src, tensor_t* dst) {
    return UNSUPPORTED_DTYPE;
};  
 
using Kernel = status(*)( const tensor_t*, tensor_t*);      
 
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

constexpr auto dispatchGELUKernel = []() { 
    std::array<Kernel, index(TYPES)> table; 
    table.fill(launchDefaultKernel);
    table[index(float32)] = launchActvKernel<float, float, GELU>;
    table[index(float64)] = launchActvKernel<double, double, GELU>;
    return table;
}();


} namespace cpu::nn {
 
status relu(const tensor_t* src, tensor_t* dst) {    
    return dispatchReLUKernel[index(src->dtype)](src, dst);
};

status silu(const tensor_t* src, tensor_t* dst) { 
    return dispatchSiLUKernel[index(src->dtype)](src, dst); 
};

status gelu(const tensor_t* src, tensor_t* dst) { 
    return dispatchGELUKernel[index(src->dtype)](src, dst); 
};
  
} // namespace cpu 