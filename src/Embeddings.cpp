#include <cstring>
#include <tannic/Callback.hpp>
#include <cstring>
#include "Embeddings.hpp"

#include "cpu/emb.hpp"
#ifdef CUDA
#include "cuda/emb.cuh"
#include "cuda/mem.cuh"
#else   
namespace cuda::nn {
status embed(const tensor_t*, const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
inline void copyFromHost(const device_t*, std::byte const* src, std::byte* dst, size_t size) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); } 
} // namespace cuda
#endif

namespace tannic::nn {

void Embedding::forward(Tensor& result, Tensor const& indexes) const {      
    Callback callback(cpu::nn::embed, cuda::nn::embed);
    callback(indexes, weight_.forward(), result);
}

void Embedding::forward(Tensor& result, std::vector<int64_t> const& lookup) const {     
    Tensor weight = weight_.forward();
    Tensor indexes(int64, {lookup.size()});  
    indexes.initialize(weight.allocator()); 

    if (std::holds_alternative<Host>(indexes.allocator())) {
        std::memcpy((void*)(indexes.bytes()), (const void*)(lookup.data()), indexes.nbytes());
    }

    else {
        Device const& resource = std::get<Device>(indexes.allocator());
        device_t dvc{.id = resource.id(), .traits = resource.blocking() ? SYNC : ASYNC}; 
        cuda::nn::copyFromHost(&dvc, (const void*)(lookup.data()),(void*)(indexes.bytes()), indexes.nbytes());
    }

    Callback callback(cpu::nn::embed, cuda::nn::embed);
    callback(indexes, weight, result);
}

}