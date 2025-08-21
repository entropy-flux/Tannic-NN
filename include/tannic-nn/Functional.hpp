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
#include <tannic/Concepts.hpp>
#include <tannic/Functions.hpp>

namespace tannic::nn {

using tannic::function::Function;

namespace functional  {  

struct ReLU {
    void operator()(Tensor const&, Tensor&) const;
};

struct SiLU {
    void operator()(Tensor const&, Tensor&) const;
};

struct GELU { 
    void operator()(Tensor const&, Tensor&) const;
};

struct Softmax {
    int axis;
    void operator()(Tensor const&, Tensor&) const;
};  
 
template<Expression Aggregation>
constexpr auto relu(Aggregation&& aggregation) {
    return Function<ReLU, Aggregation>({}, std::forward<Aggregation>(aggregation));
} 

template<Expression Aggregation>
constexpr auto silu(Aggregation&& aggregation) {
    return Function<SiLU, Aggregation>({}, std::forward<Aggregation>(aggregation));
} 

template<Expression Aggregation>
constexpr auto gelu(Aggregation&& aggregation) {
    return Function<GELU, Aggregation>({}, std::forward<Aggregation>(aggregation));
} 

template<Expression Aggregation>
constexpr auto softmax(Aggregation&& aggregation, int axis) {
    return Function<Softmax, Aggregation>({indexing::normalize(axis, aggregation.shape().rank())}, std::forward<Aggregation>(aggregation));
}  

} // namespace functional 
 
using functional::relu;
using functional::silu;
using functional::gelu; 
using functional::softmax;  

} // namespace tannic::nn



#endif // NN_FUNCTIONS_HPP