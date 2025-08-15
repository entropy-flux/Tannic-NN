#include <cstdint>
#include <iostream>
#include <cstring>
#include <tannic.hpp>
#include "server.hpp"
 
#pragma pack(push, 1)
struct Metadata {
    uint8_t  dcode;    
    size_t   offset;  
    size_t   nbytes; 
    uint8_t  rank;   
};
#pragma pack(pop) 

using namespace tannic;

void handler(std::shared_ptr<Buffer> buffer) { 
    const Metadata* metadata = reinterpret_cast<const Metadata*>(buffer->address()); 
    const char* payload = reinterpret_cast<const char*>(buffer->address());
 
    const size_t* sizes = reinterpret_cast<const size_t*>(payload + sizeof(Metadata));
    Tensor input(dtypeof(metadata->dcode), Shape(sizes, sizes + metadata->rank), metadata->offset, std::move(buffer));
    std::cout << input << std::endl;
}
 
static Router router = {
    { 1, handler }, 
};

int main() {
    Server server(8080);
    server.add(router);
    server.run();
}
