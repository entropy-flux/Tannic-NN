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

#ifndef NN_MODULE_HPP
#define NN_MODULE_HPP
 
#include <list>
#include <memory>
#include <tannic.hpp> 
#include "Parameters.hpp"

namespace tannic::nn {

struct Module {
    template<typename Self, typename... Operands>
    auto operator()(this Self&& self, Operands&&... operands) -> decltype(auto) {
        return std::forward<Self>(self).forward(std::forward<Operands>(operands)...);
    }
}; 
  
template <class Module>
class List { 
public:
    List() = default;

    List(std::initializer_list<Module> list)
    :   modules_(list) {}

    template<typename Iterator>
    List(Iterator begin, Iterator end) 
    :   modules_(begin, end) {}
 
    void add(Module&& module) {
        modules_.push_back(module);
    }
 
private:
    std::list<Module> modules_{};
};   

struct Linear : Module {
    Parameter weight;
    Parameter bias;

    constexpr Linear(type dtype, size_t input_features, size_t output_features)
    :   weight(dtype, {output_features ,input_features})
    ,   bias(dtype, {output_features})
    {}
    
    Tensor forward(Tensor input) const { 
        return matmul(input, transpose(weight, -1, -2)) + bias; // TODO: Add template specialization for this.
    }

    void initialize(std::string const& name, Parameters& parameters) const {  
        weight.initialize(name + ".weight", parameters);
        bias.initialize(name + ".bias", parameters);
    }
};  

} // tannic::nn

#endif  // NN_MODULE_HPP