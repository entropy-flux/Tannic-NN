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

#ifndef NN_FUNCTIONS_HPP
#define NN_FUNCTIONS_HPP
 
#include <tannic/Shape.hpp>
#include <tannic/Strides.hpp>
#include <tannic/Tensor.hpp>

namespace tannic::nn {

namespace expression  { 

template<class Functor, Expression Aggregation>
class Activation {
public:
    Functor functor;
    typename Trait<Aggregation>::Reference aggregation;

    constexpr Activation(Functor functor, typename Trait<Aggregation>::Reference aggregation)
    :   functor(functor)
    ,   aggregation(aggregation) 
    {} 

    constexpr auto dtype() const {
        return aggregation.dtype();
    }

    constexpr Shape const& shape() const {
        return aggregation.shape();
    }

    constexpr Strides const& strides() const {
        return aggregation.strides();
    }

    std::ptrdiff_t offset() const {
        return 0;
    }

    Tensor forward() const {
        Tensor source = aggregation.forward(); 
        Tensor target(aggregation.dtype(), aggregation.shape(), aggregation.strides());
        functor.forward(source, target);
        return target;
    }  
};

struct ReLU {
    void forward(Tensor const&, Tensor&) const;
};

struct SiLU {
    void forward(Tensor const&, Tensor&) const;
};

template<Expression Aggregation>
constexpr auto relu(Aggregation&& aggregation) {
    return Activation<ReLU, Aggregation>({}, std::forward<Aggregation>(aggregation));
} 

template<Expression Aggregation>
constexpr auto silu(Aggregation&& aggregation) {
    return Activation<SiLU, Aggregation>({}, std::forward<Aggregation>(aggregation));
}

} // namespace expression
 
using expression::relu;
using expression::silu;

} // namespace tannic::nn



#endif // NN_FUNCTIONS_HPP