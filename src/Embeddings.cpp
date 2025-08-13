#include <cstring>
#include <tannic/Callback.hpp>
#include "Embeddings.hpp"
/*
#include "cpu/emb.hpp"
#ifdef CUDA
#include "cuda/emb.cuh"
#else 
status embed(const tensor_t*, const tensor_t*, tensor_t*, stream_t)  { throw std::runtime_error("CUDA not available"); }
#endif

namespace tannic::nn {

void Embedding::forward(Tensor& result, std::vector<int64_t> const& lookup) const {    
    Callback callback(cpu::embed, cuda::embed);
//    callback(lookup, weight_.forward(), result);
}

}
*/