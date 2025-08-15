#include <iostream>
#include <tannic.hpp>
#include <tannic-nn.hpp>

using namespace tannic; 

int main() {
    std::cout << "Naive matmul MLP example with intermediate prints" << std::endl;

    // Input: 1x3
    Tensor x(float32, {1, 3});
    x.initialize();
    x[0, 0] = 1.0f;
    x[0, 1] = 2.0f;
    x[0, 2] = 3.0f;

    std::cout << "Input x:\n" << x;
 
    Tensor W1(float32, {5, 3}); W1.initialize();
    W1[0,0] = 0.1f; W1[0,1] = 0.2f; W1[0,2] = 0.3f;
    W1[1,0] = 0.4f; W1[1,1] = 0.5f; W1[1,2] = 0.6f;
    W1[2,0] = 0.7f; W1[2,1] = 0.8f; W1[2,2] = 0.9f;
    W1[3,0] = 1.0f; W1[3,1] = 1.1f; W1[3,2] = 1.2f;
    W1[4,0] = 1.3f; W1[4,1] = 1.4f; W1[4,2] = 1.5f;

    Tensor b1(float32, {5}); b1.initialize();
    b1[0] = 0.01f; b1[1] = 0.02f; b1[2] = 0.03f; b1[3] = 0.04f; b1[4] = 0.05f;

    std::cout << "Layer 1 weights W1:\n" << W1;
    std::cout << "Layer 1 bias b1:\n" << b1;
 
    Tensor W2(float32, {2, 5}); W2.initialize();
    W2[0,0] = 0.1f; W2[0,1] = 0.2f; W2[0,2] = 0.3f; W2[0,3] = 0.4f; W2[0,4] = 0.5f;
    W2[1,0] = 0.6f; W2[1,1] = 0.7f; W2[1,2] = 0.8f; W2[1,3] = 0.9f; W2[1,4] = 1.0f;

    Tensor b2(float32, {2}); b2.initialize();
    b2[0] = 0.1f; b2[1] = 0.2f;

    std::cout << "Layer 2 weights W2:\n" << W2;
    std::cout << "Layer 2 bias b2:\n" << b2;

    // Forward pass: Layer 1
    Tensor hidden = matmul(x, transpose(W1, -1, -2)) + b1;
    std::cout << "After matmul + bias (Layer 1 pre-activation):\n" << hidden;

    hidden = nn::relu(hidden);
    std::cout << "After ReLU (Layer 1 activation):\n" << hidden;

    // Forward pass: Layer 2
    Tensor output = matmul(hidden, transpose(W2, -1, -2)) + b2;
    std::cout << "After matmul + bias (Layer 2 pre-activation / output):\n" << output;
}
