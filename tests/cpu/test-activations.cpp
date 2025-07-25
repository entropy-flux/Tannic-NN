#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include "Activations.hpp"

using namespace tannic; 

class TestActivations : public ::testing::Test {
protected:  
    Tensor A;

    TestActivations() : A(float32, Shape(2, 1, 3)) { 
        A.initialize();
    }
     
    void SetUp() override { 
        float* data = reinterpret_cast<float*>(A.bytes());
        data[0] = -1.0f;
        data[1] =  0.0f;
        data[2] =  1.0f;
        data[3] = -0.5f;
        data[4] =  0.5f;
        data[5] =  2.0f;
    }
};

TEST_F(TestActivations, ReLUWorksCorrectly) {
    Tensor out = nn::relu(A);
    const float* out_data = reinterpret_cast<const float*>(out.bytes()); 
    EXPECT_FLOAT_EQ(out_data[0], 0.0f);  // -1 → 0
    EXPECT_FLOAT_EQ(out_data[1], 0.0f);  // 0 → 0
    EXPECT_FLOAT_EQ(out_data[2], 1.0f);  // 1 → 1
    EXPECT_FLOAT_EQ(out_data[3], 0.0f);  // -0.5 → 0
    EXPECT_FLOAT_EQ(out_data[4], 0.5f);  // 0.5 → 0.5
    EXPECT_FLOAT_EQ(out_data[5], 2.0f);  // 2 → 2
}

TEST_F(TestActivations, SiLUWorksCorrectly) {
    Tensor out = nn::silu(A);
    const float* out_data = reinterpret_cast<const float*>(out.bytes());

    EXPECT_NEAR(out_data[0], -1.0f / (1.0f + std::exp(1.0f)), 1e-5);
    EXPECT_NEAR(out_data[1],  0.0f,                            1e-5);
    EXPECT_NEAR(out_data[2],  1.0f / (1.0f + std::exp(-1.0f)), 1e-5);
    EXPECT_NEAR(out_data[3], -0.5f / (1.0f + std::exp(0.5f)),  1e-5);
    EXPECT_NEAR(out_data[4],  0.5f / (1.0f + std::exp(-0.5f)), 1e-5);
    EXPECT_NEAR(out_data[5],  2.0f / (1.0f + std::exp(-2.0f)), 1e-5);
} 