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
#include "Parameters.hpp"
#include "Modules.hpp"

namespace tannic::nn { 
 
class RMS : public Module {
public:
    constexpr RMS(type dtype, size_t dimension, float epsilon) 
    :   weight_(dtype, {dimension}) 
    ,   epsilon_(epsilon){}  

    Tensor forward(Tensor const& input) const {
        // TODO: Create proper fused kernel.
        return weight_ * norm(input);
    }

    void initialize(std::string const& name, Parameters& parameters) const {
        weight_.initialize(name, parameters);
    }

    Parameter const& weight() const {
        return weight_;
    }

private: 
    Tensor norm(Tensor const& input) const {
        return input * rsqrt(mean(input*input, -1, true), epsilon_);
    }

    Parameter weight_; 
    float epsilon_;
};

} // tannic::nn

#endif  // NN_NORMALIZATION_HPP