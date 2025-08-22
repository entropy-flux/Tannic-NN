// Copyright 2025 Eric Cardozo
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

#ifndef NN_NORMALIZATION_HPP
#define NN_NORMALIZATION_HPP
 
#include <list>
#include <memory>
#include <tannic.hpp> 
#include <tannic/functions.hpp>
#include <tannic/reductions.hpp>
#include "parameters.hpp"
#include "modules.hpp"

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

#endif  // NN_NORMALIZATION_HPP