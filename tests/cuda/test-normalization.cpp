#ifdef CUDA
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include <cuda_runtime.h> 
#include "parameters.hpp"
#include "modules.hpp"

using namespace tannic;
  
namespace tannic::nn { 
 
struct RMS : public Module { 
    Parameter weight;
    float epsilon;

    constexpr RMS(type dtype, size_t dimension, float epsilon) 
    :   weight(dtype, {dimension}) 
    ,   epsilon(epsilon){}  

    Tensor forward(Tensor const& input) const { 
        auto norm = input * rsqrt(mean(input*input, -1, true), epsilon);
        return weight * norm;
    }

    void initialize(std::string const& name, Parameters& parameters) const {
        weight.initialize(name, parameters);
    }
}; 

struct LayerNorm : public Module {
    Parameter weight;
    Parameter bias;
    float epsilon;
    Shape shape; 
 
    constexpr LayerNorm(type dtype, Shape shape, float epsilon = 1e-5f)
    :   weight(dtype, shape)
    ,   bias(dtype, shape)
    ,   epsilon(epsilon)
    ,   shape(shape) { 
    }

    constexpr LayerNorm(type dtype, size_t dimension, float epsilon = 1e-5f)
    :   LayerNorm(dtype, Shape(dimension), epsilon) { 
    } 

    Tensor forward(Tensor const& input) const { 
        auto mu = mean(input, -shape.rank(), true); 
        auto centered = input - mu;           
        auto squared = centered * centered;  
        auto variance = mean(squared, -shape.rank(), true);  
        auto normalized = centered * rsqrt(variance, epsilon);    
        return normalized * weight + bias; 
    }

    void initialize(std::string const& name, Parameters& parameters) const {
        weight.initialize(name + ".weight", parameters);
        bias.initialize(name + ".bias", parameters);
    }
};

} // tannic::nn


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