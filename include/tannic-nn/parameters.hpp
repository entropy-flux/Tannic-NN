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

#ifndef NN_PARAMETERS_HPP
#define NN_PARAMETERS_HPP
  
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept> 
#include <unordered_map>
#include <tannic.hpp>  
#include <tannic/context.hpp>
#include <tannic/slices.hpp>
#include <tannic/views.hpp>
#include <tannic/serialization.hpp>
  
namespace tannic::nn {

using tannic::Context;
using tannic::expression::Slice;
using tannic::expression::Transpose;

class Parameter; 

class Parameters {
public: 
    Parameters() = default; 
    void initialize(std::size_t nbytes, Environment environment = Host{});
    void initialize(std::string const& filename, Environment environment = Host{});

    std::unordered_map<std::string, std::ptrdiff_t>& offsets() {
        return offsets_;
    } 

    const std::unordered_map<std::string, std::ptrdiff_t>& offsets() const {
        return offsets_;
    }

    std::size_t nbytes() const {
        return nbytes_;
    }

    std::ptrdiff_t offset() const {
        return offset_;
    }

    Environment const& environment() const {
        return buffer_ -> environment();
    }
 

private: 
    std::size_t nbytes_ = 0;
    std::ptrdiff_t offset_ = 0;
    std::shared_ptr<Buffer> buffer_ = nullptr;
    std::unordered_map<std::string, std::ptrdiff_t> offsets_; 
    friend Parameter;
};     

class Parameter {
public:
    constexpr Parameter(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape)
    ,   strides_(shape)
    { 
        if (rank() == 0) {
            nbytes_ = dsizeof(dtype_);
        }
        else {
            std::size_t nbytes = 0; 
            for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
                nbytes += strides_[dimension] * (shape_[dimension] - 1);
            } 
            nbytes_ = (nbytes + 1) * dsizeof(dtype_);
        }
    } 

    constexpr Parameter(type dtype, Shape shape, Strides strides)
    :   dtype_(dtype)
    ,   shape_(shape)
    ,   strides_(strides)
    { 
        if (rank() == 0) {
            nbytes_ = dsizeof(dtype_);
        }
        else {
            std::size_t nbytes = 0; 
            for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
                nbytes += strides_[dimension] * (shape_[dimension] - 1);
            } 
            nbytes_ = (nbytes + 1) * dsizeof(dtype_);
        }
    } 

    constexpr uint8_t rank() const {
        return shape_.rank();
    }

    constexpr type dtype() const {
        return dtype_;
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    std::ptrdiff_t offset() const {
        return offset_;
    }

    std::size_t nbytes() const {
        return nbytes_;
    }
 
    Tensor forward(Context const& context) const {
        return Tensor(dtype_, shape_, strides_, offset_, *(std::shared_ptr<Buffer>*)(buffer_ptr_));
    }

    bool is_initialized() const {
        return buffer_ptr_ == nullptr ? false : true;
    }

public: 
    void initialize(std::string const& name, Parameters& parameters) const { 
        auto iterator = parameters.offsets().find(name);
        if (iterator == parameters.offsets().end()) {
            std::cout << "[Debug] Parameter with name " << name << " not found..." << std::endl;
            if (nbytes_ > parameters.nbytes() - parameters.offset()) {
                throw std::runtime_error("Capacity exceeded.");
            }    
            offset_ = parameters.offset_; 
            parameters.offsets_[name] = parameters.offset_; 
            parameters.offset_ += nbytes_;
            buffer_ptr_ = (void*)(&parameters.buffer_); 
            std::cout << "[Debug] Created parameter at offset: " << offset_ << std::endl;
        } else {  
            offset_ = iterator->second;
            std::cout << "[Debug] Found parameter at offset: " << offset_ << std::endl;
            buffer_ptr_ = (void*)(&parameters.buffer_); 
        } 
    }


public: 
    template<Integral Index>
    constexpr auto operator[](Index index) const {    
        return Slice<Parameter, Index>(*this, std::make_tuple(index));
    }

    constexpr auto operator[](indexing::Range range) const {  
        return Slice<Parameter, indexing::Range>(*this, std::make_tuple(range));
    } 

    template<class ... Indexes>
    constexpr auto operator[](Indexes... indexes) const { 
        return Slice<Parameter, Indexes...>(*this, std::make_tuple(indexes...));
    }  

    constexpr auto transpose(int first = -1, int second = -2) const { 
        return Transpose<Parameter>(*this, std::make_pair<int, int>(std::move(first), std::move(second)));
    }    
     
private:
    type dtype_;
    Shape shape_;
    Strides strides_; 
    mutable std::ptrdiff_t offset_ = 0;
    mutable std::size_t nbytes_ = 0;
    mutable void* buffer_ptr_ = nullptr;
};  

} namespace tannic {

#pragma pack(push, 1)
template<>
struct Metadata<nn::Parameter> {  
    uint8_t  dcode;    
    size_t   offset;  
    size_t   nbytes;  
    uint32_t namelength; 
}; 
#pragma pack(pop)  

}

#endif // NN_PARAMETERS_HPP