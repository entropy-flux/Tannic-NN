#include <tannic/tensor.hpp>
#include <tannic/callback.hpp>
#include <tannic/runtime/tensor.h>
#include <tannic/runtime/status.h>
#include <tannic/runtime/streams.h> 
#include "functional.hpp"

#include "cpu/actvs.hpp"
#include "cpu/softmax.hpp"
#ifdef CUDA 
#include "cuda/actvs.cuh"
#include "cuda/softmax.cuh"
#else
namespace cuda::nn {
using tannic::tensor_t;
using tannic::stream_t;
inline status relu(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status silu(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status gelu(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status softmax(const tensor_t*, tensor_t*, stream_t, uint8_t) { throw std::runtime_error("CUDA not available"); } 
} // namespace cuda::nn
#endif  

namespace tannic::nn::functional {  

void ReLU::operator()(Tensor const& input, Tensor& output) const {  
    Callback callback(cpu::relu, cuda::relu);
    callback(input, output);
}

void SiLU::operator()(Tensor const& input, Tensor& output) const {  
    Callback callback(cpu::silu, cuda::silu);
    callback(input, output);
}

void GELU::operator()(Tensor const& input, Tensor& output) const {  
    Callback callback(cpu::gelu, cuda::gelu);
    callback(input, output);
}

void Softmax::operator()(Tensor const& input, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::nn::softmax(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::nn::softmax(src, dst, stream, axis); }
    );
    callback(input, output);
}
 
}