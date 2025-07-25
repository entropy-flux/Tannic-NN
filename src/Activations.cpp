#include "Activations.hpp"
#include "cpu/cpu.hpp"

namespace tannic::nn::expression {

static inline tensor_t c_tensor_t(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = static_cast<void*>(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
} 
 
void ReLU::forward(Tensor const& input, Tensor& output) const { 
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output); 
    cpu::relu(&src, &dst);
}

void SiLU::forward(Tensor const& input, Tensor& output) const { 
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output); 
    cpu::silu(&src, &dst);
}

}