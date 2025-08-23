// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Neural Networks Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef NN_CONVOLUTIONAL_HPP
#define NN_CONVOLUTIONAL_HPP
 
#include <list>
#include <memory>
#include <tannic.hpp> 
#include <tannic/convolutions.hpp>
#include "parameters.hpp"
#include "modules.hpp"

namespace tannic::nn { 
    
template<auto Dimension> struct Convolutional; 

template<>
struct Convolutional<1> : Module {
    nn::Parameter weight;
    std::optional<nn::Parameter> bias;
    std::size_t stride;
    std::size_t padding;
     
    constexpr Convolutional(
        type dtype, 
        size_t input_channels, 
        size_t output_channels, 
        size_t kernel_size, 
        size_t stride = 1, 
        size_t padding = 0, 
        bool use_bias = true
    )
    :   weight(dtype, {output_channels, input_channels, kernel_size})
    ,   stride(stride)
    ,   padding(padding) {
        if (use_bias) {
            bias.emplace(dtype, Shape{output_channels});
        }
    }
 
    constexpr Convolutional(
        type dtype, 
        size_t input_channels,
        size_t output_channels,
        std::array<size_t,1> kernel_size,
        std::size_t stride = 1,
        std::size_t padding = 0,
        bool use_bias = true
    )
    :   weight(dtype, {output_channels, input_channels, kernel_size[0]})
    ,   stride(stride)
    ,   padding(padding) {
        if (use_bias) {
            bias.emplace(dtype, Shape{output_channels});
        }
    }

    void initialize(std::string const& name, Parameters& parameters) const {
        weight.initialize(name, parameters);
        if (bias.has_value()) 
            bias->initialize(name, parameters);
    }

    Tensor forward(Tensor input) const {
        if (bias.has_value()) 
            return convolve<1>(input, weight, stride, padding) + bias.value();
        else 
            return convolve<1>(input, weight, stride, padding);
    }
}; 

template<>
struct Convolutional<2> : Module {
    nn::Parameter weight;
    std::optional<nn::Parameter> bias;
    std::array<std::size_t, 2> strides;
    std::array<std::size_t, 2> padding;
    
    constexpr Convolutional(
        type dtype, 
        size_t input_channels, 
        size_t output_channels, 
        size_t kernel_size, 
        size_t stride, 
        size_t padding, 
        bool use_bias
    )
    :   weight(dtype, {output_channels, input_channels, kernel_size, kernel_size}) 
    ,   strides{stride}
    ,   padding{padding} {
        if (use_bias) {
            bias.emplace(dtype, Shape{output_channels});
        }
    } 

    constexpr Convolutional(
        type dtype, 
        size_t input_channels,
        size_t output_channels,
        std::array<size_t,2> kernel_size,
        std::array<size_t,2> strides = {1,1},
        std::array<size_t,2> padding = {0,0},
        bool use_bias = true
    )
    :   weight(dtype, {output_channels, input_channels, kernel_size[0], kernel_size[1]})
    ,   strides(strides)
    ,   padding(padding) {
        if (use_bias) {
            bias.emplace(dtype, Shape{output_channels});
        }
    } 

    void initialize(std::string const& name, Parameters& parameters) const {
        weight.initialize(name, parameters);
        if (bias.has_value()) 
            bias->initialize(name, parameters);
    }

    Tensor forward(Tensor input) const {
        if (bias.has_value()) 
            return convolve<2>(input, weight, strides, padding) + bias.value();
        else 
            return convolve<2>(input, weight, strides, padding);
            
    }
};

} // tannic::nn

#endif  // NN_CONVOLUTIONAL_HPP