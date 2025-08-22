// Copyright 2025 Eric Cardozo
//
// This file is part of the Tannic Tensor Library.
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

#ifndef NN_EMBEDDINGS_HPP
#define NN_EMBEDDINGS_HPP

#include <tannic.hpp>  
#include <tannic/Slices.hpp>
#include "Parameters.hpp"
#include "Modules.hpp"

namespace tannic::nn {

using tannic::expression::Slice;

class Embedding : public Module {
public: 

    constexpr Embedding(type dtype, size_t lenght, size_t dimension) 
    :   weight_(dtype, Shape(lenght, dimension)) 
    {}

    void initialize(std::string const& name, Parameters& parameters) const { 
        weight_.initialize(name, parameters);
        allocator_ = parameters.allocator();
    }

    template<class... Indexes>
    Tensor forward(Indexes... indexes) const {     
        std::vector<int64_t> lookup{ static_cast<int64_t>(indexes)... }; 
        Tensor result(dtype(), Shape(sizeof...(indexes), weight_.shape().back())); 
        forward(result, lookup);
        return result;
    }

    Tensor forward(Tensor const& indexes) const { 
        assert(indexes.rank() == 1 && "Indexes should be a vector"); 
        Tensor result(dtype(), Shape(indexes.shape().front(), weight_.shape().back())); 
        forward(result, indexes);
        return result;
    }

    constexpr type dtype() const { 
        return weight_.dtype(); 
    } 

    Parameter const& weight() const { 
        return weight_; 
    }

protected:
    void forward(Tensor& result, std::vector<int64_t> const& lookup) const; 
    void forward(Tensor& result, Tensor const& lookup) const;
 

public: 
    template<Integral Index>
    constexpr auto operator[](Index index) const {    
        return Slice<Parameter, Index>(weight_, std::make_tuple(index));
    }

    constexpr auto operator[](indexing::Range range) const {  
        return Slice<Parameter, indexing::Range>(weight_, std::make_tuple(range));
    } 

    template<class ... Indexes>
    constexpr auto operator[](Indexes... indexes) const { 
        return Slice<Parameter, Indexes...>(weight_, std::make_tuple(indexes...));
    }  


private:  
    Parameter weight_; 
    mutable Allocator allocator_ = Host{};
};

} // namespace tannic

#endif