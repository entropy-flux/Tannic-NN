#include <iostream>  
#include <fstream> 
#include <tannic/serialization.hpp>
#include "parameters.hpp"  
#ifdef CUDA
#include "cuda/mem.cuh"
#else 
namespace cuda::nn {
inline void copyFromHost(const device_t*, std::byte const* src, std::byte* dst, size_t size) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); }
inline bool compareFromHost(const device_t*, std::byte const* src, std::byte const* dst, size_t size) { throw std::runtime_error("CUDA compareFromHost called without CUDA support"); }
} // namespace cuda
#endif

namespace tannic::nn {   
 
using tannic::Header;
using tannic::MAGIC;
using tannic::Metadata; 

void Parameters::initialize(std::size_t nbytes, Environment environment) {  
    offset_ = 0;
    nbytes_ = nbytes;
    buffer_ = std::make_shared<Buffer>(nbytes, environment);
    offsets_.clear(); 
}

void Parameters::initialize(std::string const& filename, Environment environment) { 
    std::ifstream file;      
    Header header;  

    std::cout << "[DEBUG] Loading metadata from file: " << filename << ".metadata.tannic\n"; 
    file.open(filename + ".metadata.tannic", std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open metadata file.");
    }
 
    file.read((char*)(&header), sizeof(Header));
    if (!file) {
        throw std::runtime_error("Failed to read the header from metadata file.");
    }
    
    if (header.magic != MAGIC) {
        throw std::runtime_error("Invalid file format (bad magic number).");
    } 

    while (file.peek() != EOF) {
        Metadata<nn::Parameter> metadata;
        file.read((char*)(&metadata), sizeof(Metadata<nn::Parameter>));
        std::size_t namelength = metadata.namelength;
        std::string name(namelength, '\0');  
        std::cout << "[DEBUG] Reading parameter: " << name << "\n";
        file.read(name.data(), namelength);  
        offsets_[std::move(name)] = static_cast<std::ptrdiff_t>(metadata.offset);          
        std::cout << "[DEBUG] offset=" << metadata.offset << ", nbytes=" << metadata.nbytes << "\n";
    }

    file.close();

    file.open(filename + ".tannic", std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open data file.");
    } 
 
    file.read((char*)(&header), sizeof(Header));
    if (!file) {
        throw std::runtime_error("Failed to read the header from metadata file.");
    }

    buffer_ = std::make_shared<Buffer>(header.nbytes, environment);
    std::cout << "[DEBUG] Buffer allocated for tensor data. nbytes=" << header.nbytes << "\n";

    if (std::holds_alternative<Host>(environment)) { 
        std::cout << "[DEBUG] Environment is Host. Reading data directly.\n";
        file.read(static_cast<char*>(buffer_->address()), header.nbytes);
        if (!file) {
            throw std::runtime_error("Failed to read tensor data.");    
        }
        std::cout << "[DEBUG] Tensor data read into host buffer.\n";
    } 
    else { 
        Device const& resource = std::get<Device>(environment);
        std::vector<char> buffer(header.nbytes);
        std::cout << "[DEBUG] Environment is Device. Reading data into host temp buffer.\n";
        file.read(buffer.data(), header.nbytes);
        if (!file) {
            throw std::runtime_error("Failed to read tensor data into host buffer.");
        }

        device_t dvc{
            .id = resource.id(), 
            .traits = resource.blocking() ? SYNC : ASYNC
        };
        std::cout << "[DEBUG] Copying data from host to device id=" << resource.id() << "\n";
        cuda::nn::copyFromHost(&dvc, buffer.data(), buffer_->address(), header.nbytes);
        std::cout << "[DEBUG] Data copied to device memory.\n";
    } 
    std::cout << "[DEBUG] Parameters::initialize finished successfully.\n"; 
    file.close();
};
  
}