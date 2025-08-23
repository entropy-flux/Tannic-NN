#include <cstdint>
#include <iostream>
#include <cstring>
#include <memory>
#include <tannic.hpp>
#include <tannic/reductions.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn.hpp> 
#include <tannic-nn/functional.hpp> 
#include <tannic-nn/convolutional.hpp>
#include "server.hpp" 

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
 

// constexpr MLP model(float32, 784, 512, 10);
constexpr CNN model(float32, 1, 32, 10);

int main() {  
    nn::Parameters parameters; 
//    parameters.initialize("./examples/mnist/data/MLP");
    parameters.initialize("./examples/mnist/data/CNN");
    model.initialize(parameters);

    Server server(8080);
    while (true) {
        Socket socket = server.accept();  

        try {
            while (true) {
                Header header{};
                if (!server.read(socket, &header, sizeof(Header))) {
                    std::cout << "Client disconnected.\n";
                    break; // clean exit instead of throwing
                }

                if (header.magic != MAGIC) {
                    std::cerr << "Invalid magic! Closing connection.\n";
                    break;
                }

                Metadata<Tensor> metadata{};  
                if (!server.read(socket, &metadata, sizeof(Metadata<Tensor>))) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Shape shape; 
                size_t size;
                for (uint8_t dimension = 0; dimension < metadata.rank; dimension++) {
                    if (!server.read(socket, &size, sizeof(size_t))) {
                        std::cout << "Client disconnected.\n";
                        break;
                    }
                    shape.expand(size);
                }

                std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(metadata.nbytes);
                if (!server.read(socket, buffer->address(), metadata.nbytes)) {
                    std::cout << "Client disconnected.\n";
                    break;
                }

                Tensor input(dtypeof(metadata.dcode), shape, 0, buffer);  
                Tensor output = argmax(model(input)); 

                header = headerof(output);
                metadata = metadataof(output);

                server.write(socket, &header, sizeof(Header));
                server.write(socket, &metadata, sizeof(Metadata<Tensor>)); 
                server.write(socket, output.shape().address(), output.shape().rank() * sizeof(size_t));
                server.write(socket, output.bytes(), output.nbytes());
            }

        } catch (const std::exception& e) {
            std::cerr << "Unexpected client error: " << e.what() << "\n";
        }
    } 
}