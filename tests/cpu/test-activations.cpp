#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include "Activations.hpp"

using namespace tannic; 
 
TEST(TestActivations, TestReLU) {
    Tensor A(float32, {2,1,3}); 
    A.initialize();
    A[0][0][0] = -1.0;
    A[0][0][1] =  0.0;
    A[0][0][2] =  1.0;
    A[1][0][0] = -0.5;
    A[1][0][1] =  0.5;
    A[1][0][2] =  2.0;

    Tensor result = nn::relu(A);
    float* data = reinterpret_cast<float*>(result.bytes());

    // Expected values after ReLU
    std::vector<float> expected = {0.0, 0.0, 1.0, 0.0, 0.5, 2.0};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}

TEST(TestActivations, TestSiLU) {  
    Tensor A(float32, {2,1,3}); 
    A.initialize();
    A[0][0][0] = -1.0;
    A[0][0][1] =  0.0;
    A[0][0][2] =  1.0;
    A[1][0][0] = -0.5;
    A[1][0][1] =  0.5;
    A[1][0][2] =  2.0;

    Tensor result = nn::silu(A); 
    float* data = reinterpret_cast<float*>(result.bytes());

    // Expected values after SiLU (x * sigmoid(x))
    std::vector<float> expected;
    for (float x : {-1.0, 0.0, 1.0, -0.5, 0.5, 2.0}) {
        expected.push_back(x / (1.0 + std::exp(-x)));
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], expected[i]);
    }
}