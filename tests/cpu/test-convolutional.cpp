#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <tannic.hpp>
#include "Convolutional.hpp"

using namespace tannic;

TEST(TestConvolutional, ForwardNoBias) { 
    nn::Convolutional<2> conv(float32, 1, 1, {2,2}, {1,1}, {0,0}, false);
    nn::Parameters parameters;
    parameters.initialize(dsizeof(float32) * 100);
    conv.initialize("conv2d_no_bias", parameters);
 
    auto* w = reinterpret_cast<float*>(conv.weight.forward().bytes());
    for (size_t i = 0; i < conv.weight.forward().nbytes() /dsizeof(float32); ++i) {
        w[i] = 1.0f;
    }
 
    Tensor X(float32, {1,1,3,3});
    X.initialize();
    float* x_data = reinterpret_cast<float*>(X.bytes());
    for (int i = 0; i < 9; ++i) {
        x_data[i] = static_cast<float>(i+1);  
    }

    Tensor Y = conv.forward(X);
    float* y_data = reinterpret_cast<float*>(Y.bytes());
 
    std::vector<float> expected = {12.0f, 16.0f, 24.0f, 28.0f};

    ASSERT_EQ(Y.shape()[0], 1); // batch
    ASSERT_EQ(Y.shape()[1], 1); // channel
    ASSERT_EQ(Y.shape()[2], 2); // H
    ASSERT_EQ(Y.shape()[3], 2); // W

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 1e-5);
    }
}

TEST(TestConvolutional, ForwardWithBias) { 
    nn::Convolutional<2> conv(float32, 1, 1, {2,2}, {1,1}, {0,0}, true);
    nn::Parameters parameters;
    parameters.initialize(dsizeof(float32) * 100);
    conv.initialize("conv2d_bias", parameters);
 
    auto* w = reinterpret_cast<float*>(conv.weight.forward().bytes());
    for (size_t i = 0; i < conv.weight.nbytes() / dsizeof(float32); ++i) {
        w[i] = 1.0f;
    }
 
    auto* b = reinterpret_cast<float*>(conv.bias->forward().bytes());
    b[0] = 1.0f;
 
    Tensor X(float32, {1,1,3,3});
    X.initialize();
    float* x_data = reinterpret_cast<float*>(X.bytes());
    for (int i = 0; i < 9; ++i) {
        x_data[i] = static_cast<float>(i+1);
    }

    Tensor Y = conv.forward(X);
    float* y_data = reinterpret_cast<float*>(Y.bytes());
 
    std::vector<float> expected = {13.0f, 17.0f, 25.0f, 29.0f};

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 1e-5);
    }
}
