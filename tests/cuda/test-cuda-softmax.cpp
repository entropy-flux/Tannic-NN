#ifdef CUDA
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "Tensor.hpp"
#include "Functional.hpp"

using namespace tannic;

class TestCUDAOperationsSoftmax : public ::testing::Test {
protected:
    void SetUp() override {}
    
    void compareWithExpected(const Tensor& result, const float* expected, int size) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float cpu_data[size];
        cudaMemcpy(cpu_data, gpu_data, size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; ++i) {
            EXPECT_NEAR(cpu_data[i], expected[i], 1e-4);
        }
    }
};

TEST_F(TestCUDAOperationsSoftmax, Softmax1D) {
    Tensor logits(float32, {3});
    logits.initialize(Device(0));
    float data[3] = {2.0f, 1.0f, 0.1f};
    cudaMemcpy(logits.bytes(), data, 3 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor result = nn::softmax(logits, 0);

    const float expected[3] = {0.6590f, 0.2424f, 0.0986f};
    compareWithExpected(result, expected, 3);
}

TEST_F(TestCUDAOperationsSoftmax, Softmax2DDim1) {
    Tensor logits(float32, {2,3});
    logits.initialize(Device(0));
    float data[6] = {2.0f,1.0f,0.1f, 1.0f,3.0f,0.2f};
    cudaMemcpy(logits.bytes(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor result = nn::softmax(logits, 1);

    const float expected[6] = {
        0.6590f, 0.2424f, 0.0986f,
        0.1131f, 0.8360f, 0.0508f
    };
    compareWithExpected(result, expected, 6);
}

TEST_F(TestCUDAOperationsSoftmax, Softmax2DDim0) {
    Tensor logits(float32, {2,3});
    logits.initialize(Device(0));
    float data[6] = {2.0f,1.0f,0.1f, 1.0f,3.0f,0.2f};
    cudaMemcpy(logits.bytes(), data, 6 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor result = nn::softmax(logits, 0);

    const float expected[6] = {
        0.7311f, 0.1192f, 0.4750f,
        0.2689f, 0.8808f, 0.5250f
    };
    compareWithExpected(result, expected, 6);
}

TEST_F(TestCUDAOperationsSoftmax, Softmax3DDim2) {
    Tensor logits(float32, {2,3,4});
    logits.initialize(Device(0));
    float data[24] = {
        2.0f,1.0f,0.5f,0.1f,
        1.0f,3.0f,0.2f,0.3f,
        0.2f,0.1f,2.0f,1.0f,
        1.5f,0.5f,0.3f,2.0f,
        0.5f,1.5f,2.0f,0.1f,
        2.0f,1.0f,0.5f,0.3f
    };
    cudaMemcpy(logits.bytes(), data, 24 * sizeof(float), cudaMemcpyHostToDevice);

    Tensor result = nn::softmax(logits, 2);

    const float expected[24] = {
        0.5745f,0.2114f,0.1282f,0.0859f,
        0.1071f,0.7915f,0.0481f,0.0532f,
        0.0982f,0.0889f,0.5943f,0.2186f,
        0.3014f,0.1109f,0.0908f,0.4969f,
        0.1127f,0.3064f,0.5052f,0.0756f,
        0.5638f,0.2074f,0.1258f,0.1030f
    };
    compareWithExpected(result, expected, 24);
}

#endif
