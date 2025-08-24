#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include <cuda_runtime.h>
#include "normalization.hpp"

using namespace tannic;
 

TEST(TestCUDANormalization, TestRMS_CUDA) {
    nn::RMS norm(float32, 3, 1e-6);
    nn::Parameters parameters;
    parameters.initialize(dsizeof(float32) * 10, Device());
    norm.initialize("rms_norm", parameters);
    norm.weight[0] = 1;
    norm.weight[1] = 1;
    norm.weight[2] = 1;

    Tensor X(float32, {2,2,3}); X.initialize(Device());
    X[0, 0, 0] = 1.0;
    X[0, 0, 1] = 2.0;
    X[0, 0, 2] = 3.0;

    X[0, 1, 0] = 4.0;
    X[0, 1, 1] = 5.0;
    X[0, 1, 2] = 6.0;

    X[1, 0, 0] = -1.0;
    X[1, 0, 1] = -2.0;
    X[1, 0, 2] = -3.0;

    X[1, 1, 0] = 0.5;
    X[1, 1, 1] = 1.0;
    X[1, 1, 2] = 1.5;

    Tensor Y = norm(X);
 
    std::vector<float> y_data(2*2*3);
    cudaMemcpy(y_data.data(), Y.bytes(), 2*2*3 * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> expected = {
        0.4629f, 0.9258f, 1.3887f,
        0.7895f, 0.9869f, 1.1843f,
        -0.4629f, -0.9258f, -1.3887f,
        0.4629f, 0.9258f, 1.3887f
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 0.001);
    }
} 


TEST(TestCUDALayerNorm, TestLayerNorm_CUDA) {
    nn::LayerNorm norm(float32, 3, 1e-5f);
    nn::Parameters parameters;
    parameters.initialize(sizeof(float) * 10, Device());
 
    norm.initialize("layer_norm", parameters);
 
    norm.weight[0] = 0.5f;
    norm.weight[1] = 1.0f;
    norm.weight[2] = 1.5f;

    norm.bias[0] = 0.0f;
    norm.bias[1] = 0.1f;
    norm.bias[2] = 0.2f;
 
    Tensor X(float32, {2, 3});
    X.initialize(Device());
    X[0, 0] = 1.0f; X[0, 1] = 2.0f; X[0, 2] = 3.0f;
    X[1, 0] = 4.0f; X[1, 1] = 5.0f; X[1, 2] = 6.0f;
 
    Tensor Y = norm.forward(X);
 
    std::vector<float> y_data(2*3);
    cudaMemcpy(y_data.data(), Y.bytes(), 2*3*sizeof(float), cudaMemcpyDeviceToHost);
 
    std::vector<float> expected = {
        -0.6124f, 0.1000f, 2.0371f,
        -0.6124f, 0.1000f, 2.0371f
    };
 
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 1e-4);
    }
}
#endif