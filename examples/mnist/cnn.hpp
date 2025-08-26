#pragma once
#include <tannic.hpp>  
#include <tannic-nn.hpp> 
#include <tannic-nn/functional.hpp>   
#include <tannic-nn/convolutional.hpp>
 
using namespace tannic;  
 
struct CNN : nn::Module {
    nn::Convolutional<2> convolutional_layer;
    nn::Linear output_layer;

    constexpr CNN(type dtype, size_t input_channels, size_t hidden_channels, size_t output_size, bool bias = true) 
    :   convolutional_layer(dtype, input_channels, hidden_channels, 3, 1, 1, true)
    ,   output_layer(dtype, hidden_channels * 28 * 28, output_size, true) {}

    void initialize(nn::Parameters& parameters) const {
        convolutional_layer.initialize("convolutional_layer", parameters);
        output_layer.initialize("output_layer", parameters);
    }

    Tensor forward(Tensor features) const {    
        features = features.view(features.size(0), 1, 28, 28);  
        features = convolutional_layer(features);   
        features = nn::relu(features);  
        features = features.view(features.size(0), 32 * 28 * 28);   
        return output_layer(features);
    }
};
 