
#include <tannic.hpp>
#include <tannic-nn.hpp>

using namespace tannic;

struct MLP : nn::Module {
    nn::Linear input_layer; 
    nn::Linear output_layer;

    constexpr MLP(type dtype, size_t input_features, size_t hidden_features, size_t output_features) 
    :   input_layer(dtype, input_features, hidden_features) 
    ,   output_layer(dtype, hidden_features, output_features)
    {}

    Tensor forward(Tensor features) const {  
        features = nn::relu(input_layer(features)); 
        return output_layer(features); 
    }

    void initialize() const {
        input_layer.initialize("input_layer"); 
        output_layer.initialize("output_layer");
    }
};