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

#ifndef NN_PARAMETERS_HPP
#define NN_PARAMETERS_HPP

#include <iostream>  
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept> 
#include <unordered_map>
#include <tannic.hpp> 
#include <filesystem>

namespace tannic::nn {
 
class Parameter;

class Parameters {
public:
    static Parameters& instance() {
        static Parameters instance;
        return instance;
    }

    static void initialize(std::string const& filename) {        
        std::ifstream metadata(filename + ".metadata.cnet", std::ios::binary);
        if (!metadata)
            throw std::runtime_error("Could not open metadata file.");

        char magic[4]; metadata.read(magic, 4);
        if (std::string(magic, 4) != "CNET")
            throw std::runtime_error("Invalid magic header");    
        
        metadata.ignore(32 - 4);
        auto& instance = Parameters::instance(); 

        std::size_t offset = 0;
        std::size_t nbytes = 0;

        while (metadata.peek() != EOF) {
            uint8_t namelength; metadata.read(reinterpret_cast<char*>(&namelength), 1); 
            std::string name(namelength, '\0'); metadata.read(name.data(), namelength);    
            std::cout << "Name: "<< name << "---";
            
            metadata.read(reinterpret_cast<char*>(&offset), sizeof(size_t));
            metadata.read(reinterpret_cast<char*>(&nbytes), sizeof(size_t));  
            instance.offsets_[std::move(name)] = static_cast<std::ptrdiff_t>(offset);            

            uint8_t type = 0; metadata.read(reinterpret_cast<char*>(&type), 1);
            uint8_t rank = 0; metadata.read(reinterpret_cast<char*>(&rank), 1); 
            metadata.ignore(static_cast<std::streamsize>(rank) * 2 * sizeof(size_t)); 
            std::cout << "Offset: " << offset << "---";
            std::cout << "NBytes; " << nbytes << std::endl;
        }
   
        std::ifstream data(filename + ".cnet", std::ios::binary);
        if (!data)
            throw std::runtime_error("Could not open tensor data file.");


        std::cout << "Offset: " << offset << " ";
        std::cout << "NBytes; " << nbytes << std::endl;

        std::size_t size = offset + nbytes; 
        instance.buffer_ = std::make_shared<Buffer>(size);
        data.read(static_cast<char*>(instance.buffer_->address()), size);
        if (!data)
            throw std::runtime_error("Failed to read tensor data.");    
    }  

private:
    Parameters() = default;
    ~Parameters() = default;
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete; 
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
    {}

    void initialize(std::string const& name) const { 
        auto& parameters = Parameters::instance();
        auto iterator = parameters.offsets_.find(name);
        if (iterator == parameters.offsets_.end()) {
            throw std::runtime_error("Parameter name not found: " + name);
        }
        offset_ = iterator->second;
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

    Tensor forward() const {
        auto& parameters = Parameters::instance();
        return Tensor(dtype_, shape_, strides_, offset_, parameters.buffer_);
    }

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
    mutable std::ptrdiff_t offset_ = 0;
    mutable std::size_t nbytes_ = 0;
}; 

} // tannic::nn
 
#endif // NN_PARAMETERS_HPP