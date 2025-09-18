#include <cstring>
#include <tannic/callback.hpp>
#include <cstring>
#include "embeddings.hpp"

#include "cpu/emb.hpp"
#ifdef CUDA
#include "cuda/emb.cuh"
#include "cuda/mem.cuh"
#else   
namespace cuda::nn {
inline status embed(const tensor_t*, const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); } 
inline void copyFromHost(const device_t*, const void* /*host memory address*/, void* /*device memory address*/, size_t /*number of bytes*/) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); } 
} // namespace cuda
#endif

namespace tannic::nn {

void Embedding::forward(Tensor& result, Tensor const& indexes) const {      
    Callback callback(cpu::nn::embed, cuda::nn::embed);
    Context context{};
    callback(indexes, weight_.forward(context), result);
}

void Embedding::forward(Tensor& result, std::vector<int64_t> const& lookup) const {     
    Context context{};
    Tensor weight = weight_.forward(context);
    Tensor indexes(int64, {lookup.size()});  
    indexes.initialize(weight.environment()); 

    if (std::holds_alternative<Host>(indexes.environment())) {
        std::memcpy((void*)(indexes.bytes()), (const void*)(lookup.data()), indexes.nbytes());
    }

    else {
        Device const& resource = std::get<Device>(indexes.environment());
        device_t dvc{.id = resource.id()}; 
        cuda::nn::copyFromHost(&dvc, static_cast<const void*>(lookup.data()), static_cast<void*>(indexes.bytes()), indexes.nbytes()); 
    }

    Callback callback(cpu::nn::embed, cuda::nn::embed);
    callback(indexes, weight, result);
}

}