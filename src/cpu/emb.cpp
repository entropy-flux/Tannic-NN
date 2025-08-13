#include "emb.hpp"

namespace {

template<typename I, typename D>
void stridedmbeddingKernel(
    const I* indexes_ptr, const shape_t& indexes_shape, const strides_t& indexes_strides,
    const D* embedding_ptr, const shape_t& embedding_shape, const strides_t& embedding_strides,
    D* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank, size_t num_indexes
) {
    size_t cnt[8] = {0};   
    for (size_t idx = 0; idx < num_indexes; ++idx) { 
        size_t idx_offset = 0;
        for (uint8_t dim = 0; dim < rank; ++dim) {
            size_t coord = (indexes_shape.sizes[dim] == 1) ? 0 : cnt[dim];
            idx_offset += coord * indexes_strides.sizes[dim];
        }

        I index = indexes_ptr[idx_offset];  
        for (size_t d = 0; d < embedding_shape.sizes[1]; ++d) {
            dst_ptr[idx * embedding_shape.sizes[1] + d] = embedding_ptr[index * embedding_strides.sizes[0] + d];
        }
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < indexes_shape.sizes[dim]) break;
            cnt[dim] = 0;
        }
    }
}

template<typename D>
status launchEmbeddingKernel(const tensor_t* indices, const tensor_t* embedding, tensor_t* dst) {  
    size_t num_indices = 1;
    for (uint8_t i = 0; i < indices->rank; ++i) {
        num_indices *= indices->shape.sizes[i];
    }

    stridedmbeddingKernel<int64_t, D>(
        (const int64_t*)indices->address, indices->shape, indices->strides,
        (const D*)embedding->address, embedding->shape, embedding->strides,
        (D*)dst->address, dst->shape, dst->strides,
        indices->rank, num_indices
    );  
    return SUCCESS;
}

} namespace cpu {

status embed(const tensor_t* src_indices, const tensor_t* embedding_matrix, tensor_t* dst) {
    if (src_indices->dtype != int64) {
        return UNSUPPORTED_DTYPE; 
    }
    switch (embedding_matrix->dtype) {
        case float32: return launchEmbeddingKernel<float> (src_indices, embedding_matrix, dst);
        case float64: return launchEmbeddingKernel<double>(src_indices, embedding_matrix, dst);
        default:
            return UNSUPPORTED_DTYPE;
    }
} 
    
}