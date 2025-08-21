#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include "Normalization.hpp"

using namespace tannic;  
TEST(TestNormalization, TestRMS) { 
    nn::RMS norm(float32, 3, 1e-6);
    nn::Parameters parameters; 
    parameters.initialize(dsizeof(float32)*10);  
    norm.initialize("rms_norm", parameters); 
    norm.weight[0] = 1;
    norm.weight[1] = 1;
    norm.weight[2] = 1;

    Tensor X(float32, {2,2,3}); 
    X.initialize(); 
    X[0, 0, 0] = 1.0; X[0, 0, 1] = 2.0; X[0, 0, 2] = 3.0;
    X[0, 1, 0] = 4.0; X[0, 1, 1] = 5.0; X[0, 1, 2] = 6.0;
    X[1, 0, 0] = -1.0; X[1, 0, 1] = -2.0; X[1, 0, 2] = -3.0;
    X[1, 1, 0] = 0.5;  X[1, 1, 1] = 1.0;  X[1, 1, 2] = 1.5;

    Tensor Y = norm(X);
    float* y_data = reinterpret_cast<float*>(Y.bytes()); 

    std::vector<float> expected = {
        0.4629,  0.9258,  1.3887,    
        0.7895,  0.9869,  1.1843, 
        -0.4629, -0.9258, -1.3887,  
        0.4629,  0.9258,  1.3887  
    }; 
  
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 0.001);
    } 
}
 
TEST(TestNormalization, TestLayerNorm) { 
    nn::LayerNorm norm(float32, 3, 1e-5f);
    nn::Parameters parameters; 
    parameters.initialize(dsizeof(float32)*10);  
    norm.initialize("layer_norm", parameters); 
 
    norm.weight[0] = 0.5f; 
    norm.weight[1] = 1.0f; 
    norm.weight[2] = 1.5f;
    norm.bias[0] = 0.0f; 
    norm.bias[1] = 0.1f; 
    norm.bias[2] = 0.2f;

    Tensor X(float32, {2, 3}); 
    X.initialize(); 
    X[0, 0] = 1.0f; X[0, 1] = 2.0f; X[0, 2] = 3.0f;
    X[1, 0] = 4.0f; X[1, 1] = 5.0f; X[1, 2] = 6.0f;

    Tensor Y = norm.forward(X);
    float* y_data = reinterpret_cast<float*>(Y.bytes()); 

    std::vector<float> expected = {
        -0.6124f, 0.1000f, 2.0371f,
        -0.6124f, 0.1000f, 2.0371f
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_data[i], expected[i], 1e-4);
    }
}

/* 
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        print("Input x:", x)

        # Step 1: square the input
        x_sq = x.pow(2)
        print("Step 1 - x squared:", x_sq)

        # Step 2: compute mean along last dimension
        mean_sq = x_sq.mean(-1, keepdim=True)
        print("Step 2 - mean of squared values:", mean_sq)

        # Step 3: add epsilon for numerical stability
        mean_sq_eps = mean_sq + self.eps
        print("Step 3 - mean + eps:", mean_sq_eps)

        # Step 4: take reciprocal square root
        rsqrt = torch.rsqrt(mean_sq_eps)
        print("Step 4 - rsqrt:", rsqrt)

        # Step 5: multiply original input by rsqrt
        normed = x * rsqrt
        print("Step 5 - normalized output:", normed)

        return normed

    def forward(self, x):
        x_norm = self._norm(x.float()).type_as(x)
        print("After type casting:", x_norm)
        output = x_norm * self.weight
        print("After applying weight:", output)
        return output


# Instantiate RMSNorm
with torch.no_grad():
  dim = 3
  norm = RMSNorm(dim) 

  # Create a 3D tensor of shape (2, 2, 3) with explicit assignments
  X = torch.zeros(2, 2, 3)

  # Assign values individually
  X[0, 0, 0] = 1.0
  X[0, 0, 1] = 2.0
  X[0, 0, 2] = 3.0

  X[0, 1, 0] = 4.0
  X[0, 1, 1] = 5.0
  X[0, 1, 2] = 6.0

  X[1, 0, 0] = -1.0
  X[1, 0, 1] = -2.0
  X[1, 0, 2] = -3.0

  X[1, 1, 0] = 0.5
  X[1, 1, 1] = 1.0
  X[1, 1, 2] = 1.5

  # Apply RMSNorm
  output = norm(X)

  print("\nRMSNorm(X):\n", output) 
*/