#include <cstdint>
#include <iostream>
#include <cstring>
#include <memory>
#include <tannic.hpp>
#include <tannic-nn.hpp>  
#include <tannic-nn/Normalization.hpp>

using namespace tannic;    

int main() {  
    nn::Parameters parameters; parameters.initialize(1024); 


/*
import torch
import torch.nn as nn
 
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
 
layer_norm = nn.LayerNorm(normalized_shape=3)
 
layer_norm.weight.data = torch.tensor([0.5, 1.0, 1.5])
layer_norm.bias.data   = torch.tensor([0.0, 0.1, 0.2])
 
output = layer_norm(x)

print("Input:\n", x)
print("Weight:\n", layer_norm.weight)
print("Bias:\n", layer_norm.bias)
print("Output:\n", output)*/

    nn::LayerNorm norm(float32, 3); norm.initialize("norm", parameters); 
    norm.weight[0] = 0.5f;
    norm.weight[1] = 1.0f;
    norm.weight[2] = 1.5f;

    norm.bias[0] = 0.0f;
    norm.bias[1] = 0.1f;
    norm.bias[2] = 0.2f;

    Tensor X = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
 
    std::cout << norm(X);
}