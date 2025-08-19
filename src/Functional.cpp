#include <tannic/Tensor.hpp>
#include <tannic/Callback.hpp>
#include <tannic/runtime/tensor.h>
#include <tannic/runtime/status.h>
#include <tannic/runtime/streams.h> 
#include "Functional.hpp"

#include "cpu/softmax.hpp"
#ifdef CUDA 
#include "cuda/softmax.cuh"
#else
namespace cuda::nn {
using tannic::tensor_t;
using tannic::stream_t;
inline status softmax(const tensor_t*, tensor_t*, stream_t, uint8_t) { throw std::runtime_error("CUDA not available"); } 
} // namespace cuda::nn
#endif  

namespace tannic::nn::functional { 
 
void Softmax::operator()(Tensor const& input, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::nn::softmax(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::nn::softmax(src, dst, stream, axis); }
    );
    callback(input, output);
}
 
}