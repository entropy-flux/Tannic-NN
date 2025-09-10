#include <iostream>
#include <tannic.hpp>
#include <tannic-nn.hpp>
#include <tannic-nn/functional.hpp>

using namespace tannic;

int main() { 
    nn::Parameters parameters; parameters.initialize("data/dev-embd");
    nn::Embedding embeddings(float32, 128, 64); embeddings.initialize(parameters);

    Tensor tokens(int64, {2, 6}); tokens.initialize({
        {19, 51, 25, 88, 115, 83},
        {55, 69, 126, 70, 99, 61}
    });  


    std::cout << embeddings(tokens) << std::endl; 
}