#include <tannic.hpp>
#include <tannic-nn.hpp>
 
#include <unistd.h>
#include <netinet/in.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>  

#include "examples/mnist/model.hpp"
#include "examples/mnist/server.hpp"

/*
Copy and paste this fail in the root folder's main.cpp
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