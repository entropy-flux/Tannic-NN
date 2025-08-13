#include <iostream>  
#include <fstream> 
#include "Parameters.hpp" 
#ifdef CUDA
#include "cuda/mem.cuh"
#else 
namespace cuda::nn {
inline void copyFromHost(const device_t*, std::byte const* src, std::byte* dst, size_t size) { throw std::runtime_error("CUDA copyFromHost called without CUDA support"); }
inline bool compareFromHost(const device_t*, std::byte const* src, std::byte const* dst, size_t size) { throw std::runtime_error("CUDA compareFromHost called without CUDA support"); }
} // namespace cuda
#endif


namespace tannic::nn {  

void Parameters::initialize(std::size_t nbytes, Allocator allocator) { 
    offset_ = 0;
    nbytes_ = nbytes;
    buffer_ = std::make_shared<Buffer>(nbytes, allocator);
    offsets_.clear();
}

void Parameters::initialize(std::string const& filename, Allocator allocator) {        
    std::ifstream metadata(filename + ".metadata.cnet", std::ios::binary);
    if (!metadata)
        throw std::runtime_error("Could not open metadata file.");

    char magic[4]; metadata.read(magic, 4);
    if (std::string(magic, 4) != "CNET")
        throw std::runtime_error("Invalid magic header");    
    
    metadata.ignore(32 - 4); 

    std::size_t offset = 0;
    std::size_t nbytes = 0;

    while (metadata.peek() != EOF) {
        uint8_t namelength; metadata.read(reinterpret_cast<char*>(&namelength), 1); 
        std::string name(namelength, '\0'); metadata.read(name.data(), namelength);    
        std::cout << "Name: "<< name << "---";
        
        metadata.read(reinterpret_cast<char*>(&offset), sizeof(size_t));
        metadata.read(reinterpret_cast<char*>(&nbytes), sizeof(size_t));  
        offsets_[std::move(name)] = static_cast<std::ptrdiff_t>(offset);            

        uint8_t type = 0; metadata.read(reinterpret_cast<char*>(&type), 1);
        uint8_t rank = 0; metadata.read(reinterpret_cast<char*>(&rank), 1); 
        metadata.ignore(static_cast<std::streamsize>(rank) * 2 * sizeof(size_t));  
    }
    nbytes += offset;   
 
    std::ifstream data(filename + ".cnet", std::ios::binary);
    if (!data)
        throw std::runtime_error("Could not open tensor data file.");  
 
    nbytes_ = nbytes;
    offset_ = nbytes;
    buffer_ = std::make_shared<Buffer>(nbytes, allocator);
    
    if (std::holds_alternative<Host>(allocator)) { 
        data.read(static_cast<char*>(buffer_->address()), nbytes);
        if (!data)
            throw std::runtime_error("Failed to read tensor data.");    
    } 

    else { 
        Device const& resource = std::get<Device>(allocator);
        std::vector<char> buffer(nbytes);
        data.read(buffer.data(), nbytes);
        if (!data)
            throw std::runtime_error("Failed to read tensor data into host buffer.");

        device_t dvc{
            .id = resource.id(), 
            .traits = resource.blocking() ? SYNC : ASYNC
        };
        cuda::nn::copyFromHost(&dvc,buffer.data(), buffer_->address(), nbytes);
    } 
}  

}