#include <cstdint>
#include <iostream>
#include <cstring>
#include <memory>
#include <tannic.hpp>
#include <tannic-nn.hpp>  

using namespace tannic;    

struct Patches : nn::Module { 
    size_t ph;
    size_t pw;
    size_t pd;
    nn::Linear transformation;  
  
    constexpr Patches(type dtype, size_t model_dimension, size_t patch_height, size_t patch_width, size_t number_of_channels) 
    :   ph(patch_height)      
    ,   pw(patch_width)
    ,   pd(patch_height * patch_width * number_of_channels)
    ,   transformation(dtype, pd, model_dimension) {}
    
    void initialize(std::string const& name, nn::Parameters& parameters) const {
        transformation.initialize(name, parameters);
    }

    Tensor forward(Tensor features) const {   
        size_t b = features.size(0);
        size_t c = features.size(1);
        size_t h = features.size(2);
        size_t w = features.size(3); 
        features = view(features, b, c, h / ph, ph, w / pw, pw); 
        features = permute(features, 0, 2, 4, 3, 5, 1); 
        features = reshape(features, b, (h / ph) * (w / pw), ph * pw * c);  
        features =  transformation(features); 
        return features;
    }
};

class CLSToken : nn::Module {
    nn::Parameter token;

    constexpr CLSToken(type dtype, size_t model_dimension)
    :   token(dtype, {1, 1, model_dimension})
    {}

    Tensor forward(Tensor input) {
        size_t batch_size = input.size(0);
        auto cls = expand(token, batch_size, -1, -1);
        return concatenate(cls, input, 1);
    }
};

Tensor attention(Tensor Q, Tensor K, Tensor V) {
    auto scale = 1 / std::sqrt(K.size(-1));
    auto score = matmul(Q, K.transpose(-2, -1), scale);
    return matmul(score, nn::softmax(score,-1));
}
 
struct Attention : nn::Module {
    nn::Parameter WQ;
    nn::Parameter WK;
    nn::Parameter WV;
    nn::Parameter WO;
    size_t heads;

    Tensor forward(Tensor sequence) const { 
        auto Q = matmul(sequence, WQ.transpose());
        auto K = matmul(sequence, WK.transpose());
        auto V = matmul(sequence, WV.transpose());
        sequence = attention(split(Q), split(K), split(V));
        return matmul(sequence, WO.transpose());
    }   

    Tensor split(Tensor sequence) const {
        size_t batch_size = sequence.size(0);
        size_t sequence_length = sequence.size(1);
        size_t model_dimension = sequence.size(2);
        sequence = sequence.view(batch_size, sequence_length, heads, model_dimension / heads);
        return sequence.transpose(1, 2); 
    }

    Tensor concat(Tensor sequence) const { 
        size_t batch_size = sequence.size(0);
        size_t number_of_heads = sequence.size(1);
        size_t sequence_lenght = sequence.size(2);
        size_t heads_dimension = sequence.size(3);
        sequence = sequence.transpose(1, 2);
        return reshape(sequence, batch_size, sequence_lenght, heads_dimension* number_of_heads);
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        WQ.initialize(name, parameters);
        WK.initialize(name, parameters);
        WV.initialize(name, parameters);
        WO.initialize(name, parameters);
    } 
};

constexpr size_t number_of_patches(size_t image_height, size_t patch_height, size_t image_width, size_t patch_width) {
    return (image_height / patch_height) * (image_width / patch_width);
} 

struct PositionalEncoding : nn::Module {
    nn::Parameter position_embeddings;
    size_t limit;

    Tensor forward(Tensor input) const { 
        if (input.size(1) > limit)
            throw Exception("input sequence is too long");          
        return input + position_embeddings[{0,-1}][{0, int(input.size(1))}];
    }
};
 

int main() {  
    nn::Parameters parameters; parameters.initialize(1024); 
}