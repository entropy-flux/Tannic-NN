/*
from struct import pack, unpack
from torch import flatten
from torch import Tensor
from torch import float32
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets.mnist import MNIST
from pytannic.client import Client
from pytannic.torch.serialization import serialize, deserialize

transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

dataset = MNIST(
    download=True,
    root='data/mnist',
    train=False,
    transform=transform
)

# Tensor input(float32, {1,3});
# input.initialize();
# input[0,0] = 1.0;
# input[0,1] = 2.0;
# input[0,2] = 3.0;

if __name__ == "__main__":
    with Client('127.0.0.1', 8080) as client:
        request = Tensor([1.0, 2.0, 3.0]).reshape((1, 3))
        print(request.stride())
        assert request.dtype == float32
        print("Request: ", request)

        serialized = serialize(request)
        client.send(serialized)

        response = client.receive()
        tensor = deserialize(response)   tensor([[9.415, 21.94]])

        print("Response:", tensor) prints tensor([1])
*/

#include <tannic.hpp>
#include <tannic-nn.hpp>   
#include "server.hpp"
 
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

    void initialize(nn::Parameters& parameters) const {
        input_layer.initialize("input_layer", parameters); 
        output_layer.initialize("output_layer", parameters);
    }
};

constexpr MLP model(float32, 3, 5, 2);

Tensor Server::forward(Tensor input) const {   
    std::cout << input << std::endl;
    Tensor output = model(input);
    std::cout << output << std::endl;
    Tensor response = argmax(output);    
    std::cout << response << std::endl;
    return response;
} 

int main() { 
    nn::Parameters parameters; parameters.initialize(32*sizeof(float));
    model.initialize(parameters);  
    model.input_layer.weight[0,0] = 0.1f; model.input_layer.weight[0,1] = 0.2f; model.input_layer.weight[0,2] = 0.3f;
    model.input_layer.weight[1,0] = 0.4f; model.input_layer.weight[1,1] = 0.5f; model.input_layer.weight[1,2] = 0.6f;
    model.input_layer.weight[2,0] = 0.7f; model.input_layer.weight[2,1] = 0.8f; model.input_layer.weight[2,2] = 0.9f;
    model.input_layer.weight[3,0] = 1.0f; model.input_layer.weight[3,1] = 1.1f; model.input_layer.weight[3,2] = 1.2f;
    model.input_layer.weight[4,0] = 1.3f; model.input_layer.weight[4,1] = 1.4f; model.input_layer.weight[4,2] = 1.5f;
 
    model.input_layer.bias[0] = 0.01f; model.input_layer.bias[1] = 0.02f; model.input_layer.bias[2] = 0.03f;
    model.input_layer.bias[3] = 0.04f; model.input_layer.bias[4] = 0.05f;
 
    model.output_layer.weight[0,0] = 0.1f; model.output_layer.weight[0,1] = 0.2f; model.output_layer.weight[0,2] = 0.3f;
    model.output_layer.weight[0,3] = 0.4f; model.output_layer.weight[0,4] = 0.5f;

    model.output_layer.weight[1,0] = 0.6f; model.output_layer.weight[1,1] = 0.7f; model.output_layer.weight[1,2] = 0.8f;
    model.output_layer.weight[1,3] = 0.9f; model.output_layer.weight[1,4] = 1.0f;
 
    model.output_layer.bias[0] = 0.1f; model.output_layer.bias[1] = 0.2f;
 

    Server server(8080);
    server.run();
    return 0;
} 


 