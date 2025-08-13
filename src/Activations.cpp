#include <tannic/Tensor.hpp>
#include <tannic/Callback.hpp>
#include <tannic/runtime/tensor.h>
#include <tannic/runtime/status.h>
#include <tannic/runtime/streams.h> 
#include "Activations.hpp"

#include "cpu/actvs.hpp"
#ifdef CUDA 
#include "cuda/actvs.cuh"
#else
inline status relu(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
inline status silu(const tensor_t*, tensor_t*, stream_t) { throw std::runtime_error("CUDA not available"); }
#endif  

namespace tannic::nn::expression { 
 
void ReLU::forward(Tensor const& input, Tensor& output) const {  
    Callback callback(cpu::relu, cuda::relu);
    callback(input, output);
}

void SiLU::forward(Tensor const& input, Tensor& output) const {  
    Callback callback(cpu::silu, cuda::silu);
    callback(input, output);
}

}