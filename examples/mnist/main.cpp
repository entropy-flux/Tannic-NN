#include <tannic.hpp>
#include <tannic-nn.hpp>   
#include "examples/mnist/server.hpp"  
 
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
 
Tensor Server::forward(Tensor input) const { 
    Tensor result = argmax(model(input)); 
    return result;
}

constexpr MLP model(float32, 784, 512, 10); 

int main() {
    nn::Parameters parameters; parameters.initialize("./examples/mnist/data/mlp");
    model.initialize(parameters);
    Server server(8080);
    server.run();
    return 0;
}