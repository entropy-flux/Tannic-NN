#include <tannic.hpp>
#include <tannic-nn.hpp>  

#include "examples/mnist/model.hpp"
#include "examples/mnist/server.hpp"

/*
Copy and paste this file in the folder's root main.cpp and run it with main.sh
Then run the python client to try it out.
*/

using namespace tannic; 

constexpr MLP model(float32, 784, 512, 10);

Tensor Server::forward(Tensor input) const { 
    Tensor result = argmax(model(input)); 
    return result;
}

int main() {
    nn::Parameters::initialize("./examples/mnist/data/mlp");
    model.initialize();
    Server server(8080);
    server.run();
    return 0;
}