#include <cstdint>
#include <iostream>
#include <cstring>
#include <memory>
#include <tannic.hpp>
#include <tannic-nn.hpp>
#include "serialization.hpp"
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

constexpr MLP model(float32, 784, 512, 10);

int main() {  
    nn::Parameters parameters; 
    parameters.initialize("./examples/mnist/data/mlp");
    model.initialize(parameters);

    Server server(8080);

    while (true) {
        Socket socket = server.accept();  

        try {
            Header header{};
            server.read(socket, &header, sizeof(Header));

            if (header.magic != magic) {
                std::cerr << "Invalid magic! Closing connection.\n";
                continue;  
            }

            Metadata metadata{};  
            server.read(socket, &metadata, sizeof(Metadata));   
            
            Shape shape; 
            size_t size;
            for (uint8_t dimension = 0; dimension < metadata.rank; dimension++) {
                server.read(socket, &size, sizeof(size_t));
                shape.expand(size);
            }   
            
            std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(metadata.nbytes);
            server.read(socket, buffer->address(), metadata.nbytes);
            
            Tensor input(dtypeof(metadata.dcode), shape, 0, buffer);  
            Tensor output = argmax(model(input)); 

            header = headerof(output);
            metadata = metadataof(output);

            server.write(socket, &header, sizeof(Header));
            server.write(socket, &metadata, sizeof(Metadata)); 
            server.write(socket, output.shape().address(), output.shape().rank() * sizeof(size_t));
            server.write(socket, output.bytes(), output.nbytes());

        } catch (const std::exception& e) {
            std::cerr << "Client error: " << e.what() << "\n";
        }
    } 
}