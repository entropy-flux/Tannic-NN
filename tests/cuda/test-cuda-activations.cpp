#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include <cuda_runtime.h> 
#include "Activations.hpp"

using namespace tannic;

class TestCUDAActivations : public ::testing::Test {
protected:
    Tensor A;

    TestCUDAActivations() : A(float32, Shape(2, 1, 3)) {
        A.initialize(Device()); 
        A[0][0][0] = -1.0;
        A[0][0][1] =  0.0;
        A[0][0][2] =  1.0;
        A[1][0][0] = -0.5;
        A[1][0][1] =  0.5;
        A[1][0][2] =  2.0;
    }
 
    void compareWithExpected(const Tensor& result, const float expected[6]) {
        float* gpu_data = reinterpret_cast<float*>(result.bytes());
        float cpu_data[6];
         
        cudaMemcpy(cpu_data, gpu_data, 6 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < 6; ++i) {
            EXPECT_FLOAT_EQ(cpu_data[i], expected[i]);
        }
    }
};


TEST_F(TestCUDAActivations, TestCUDAReLU) {
    Tensor result = nn::relu(A);
    const float expected[6] = {0.0, 0.0, 1.0, 0.0, 0.5, 2.0};
    compareWithExpected(result, expected);
}
  
TEST_F(TestCUDAActivations, TestCUDASiLU) {
    Tensor result = nn::silu(A);
    const float expected[6] = {
        -1.0f * (1.0f / (1.0f + expf(1.0f))), 
        0.0f * (1.0f / (1.0f + expf(0.0f))),    
        1.0f * (1.0f / (1.0f + expf(-1.0f))), 
        -0.5f * (1.0f / (1.0f + expf(0.5f))),   
        0.5f * (1.0f / (1.0f + expf(-0.5f))),    
        2.0f * (1.0f / (1.0f + expf(-2.0f)))    
    };
    compareWithExpected(result, expected);
} 
#endif