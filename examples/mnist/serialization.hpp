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

#pragma once 
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string> 
#include <cstdint>
#include <tannic/tensor.hpp>
#include <tannic/serialization.hpp>
#include <tannic-nn/parameters.hpp>

namespace tannic {  
  
#pragma pack(push, 1)
template<>
struct Metadata<nn::Parameter> {  
    uint8_t  dcode;    
    size_t   offset;  
    size_t   nbytes; 
    uint8_t  rank;   
    uint32_t namelength; 
}; 
#pragma pack(pop)  
 

} // namespace tannic 