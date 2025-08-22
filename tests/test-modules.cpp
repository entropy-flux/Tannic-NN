#include <gtest/gtest.h>
#include <tannic.hpp>
#include <tannic-nn.hpp>    
#include <tannic-nn/functional.hpp>    
 
using namespace tannic;

struct MLP : nn::Module {
    nn::Linear input_layer; 
    nn::Linear output_layer;

    constexpr MLP(type dtype, size_t input_features, size_t hidden_features, size_t output_features) 
    :   input_layer(dtype, input_features, hidden_features) 
    ,   output_layer(dtype, hidden_features, output_features)
    {}

    void initialize(nn::Parameters& parameters) const {
        input_layer.initialize("input_layer", parameters); 
        output_layer.initialize("output_layer", parameters);
    }

    Tensor forward(Tensor features) const {  
        features = nn::relu(input_layer(features));  
        return output_layer(features); 
    }
};

constexpr MLP model(float32, 3, 5, 2);

TEST(TestMLP, TestForward) {
    nn::Parameters parameters; parameters.initialize(32*sizeof(float));
    model.initialize(parameters); 

    // Input layer weights (5x3)
    model.input_layer.weight[0,0] = 0.1f; model.input_layer.weight[0,1] = 0.2f; model.input_layer.weight[0,2] = 0.3f;
    model.input_layer.weight[1,0] = 0.4f; model.input_layer.weight[1,1] = 0.5f; model.input_layer.weight[1,2] = 0.6f;
    model.input_layer.weight[2,0] = 0.7f; model.input_layer.weight[2,1] = 0.8f; model.input_layer.weight[2,2] = 0.9f;
    model.input_layer.weight[3,0] = 1.0f; model.input_layer.weight[3,1] = 1.1f; model.input_layer.weight[3,2] = 1.2f;
    model.input_layer.weight[4,0] = 1.3f; model.input_layer.weight[4,1] = 1.4f; model.input_layer.weight[4,2] = 1.5f;

    // Input layer biases (5)
    model.input_layer.bias.value()[0] = 0.01f; model.input_layer.bias.value()[1] = 0.02f; model.input_layer.bias.value()[2] = 0.03f;
    model.input_layer.bias.value()[3] = 0.04f; model.input_layer.bias.value()[4] = 0.05f;

    // Output layer weights (2x5)
    model.output_layer.weight[0,0] = 0.1f; model.output_layer.weight[0,1] = 0.2f; model.output_layer.weight[0,2] = 0.3f;
    model.output_layer.weight[0,3] = 0.4f; model.output_layer.weight[0,4] = 0.5f;

    model.output_layer.weight[1,0] = 0.6f; model.output_layer.weight[1,1] = 0.7f; model.output_layer.weight[1,2] = 0.8f;
    model.output_layer.weight[1,3] = 0.9f; model.output_layer.weight[1,4] = 1.0f;

    // Output layer biases (2)
    model.output_layer.bias.value()[0] = 0.1f; model.output_layer.bias.value()[1] = 0.2f;


    Tensor input(float32, {1,3}); input.initialize();
    input[0,0] = 1.0;
    input[0,1] = 2.0;
    input[0,2] = 3.0; 

    Tensor output = model(input); 

    float* data = reinterpret_cast<float*>(output.bytes());
    EXPECT_NEAR(data[0], 9.4550, 1);
    EXPECT_NEAR(data[1], 22.1300, 1);
}


/*
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.input_layer = nn.Linear(input_features, hidden_features)
        self.output_layer = nn.Linear(hidden_features, output_features)
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        # Manually fill weights for deterministic testing
        with torch.no_grad():
            # Input layer weights (hidden_features x input_features)
            self.input_layer.weight.copy_(torch.tensor([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ]))
            self.input_layer.bias.copy_(torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05]))

            # Output layer weights (output_features x hidden_features)
            self.output_layer.weight.copy_(torch.tensor([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9, 1.0],
            ]))
            self.output_layer.bias.copy_(torch.tensor([0.1, 0.2]))


# Example usage
mlp = MLP(input_features=3, hidden_features=5, output_features=2)
x = torch.tensor([[1.0, 2.0, 3.0]])
y = mlp(x)
print(y)

*/