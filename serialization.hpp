#pragma once
#include <unistd.h>
#include <netinet/in.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>     

namespace tannic {

constexpr uint32_t magic = 0x43495245;

#pragma pack(push, 1)  
struct Header {
    uint32_t magic;      
    uint8_t  protocol;
    uint8_t  code;
    uint16_t checksum;
    uint64_t nbytes; 
};
#pragma pack(pop) 
 
#pragma pack(push, 1)
struct Metadata {
    uint8_t  dcode;    
    size_t   offset;  
    size_t   nbytes; 
    uint8_t  rank;   
};
#pragma pack(pop) 

inline Header headerof(Tensor const& tensor) {
    return Header{
        .magic=0x43495245,
        .protocol=0,
        .code=1,
        .checksum=0xABCD,
        .nbytes=tensor.nbytes() + sizeof(Metadata) + tensor.rank() * sizeof(size_t)
    };
}

inline Metadata metadataof(Tensor const& tensor) {
    return Metadata {
        .dcode = dcodeof(tensor.dtype()),
        .offset = static_cast<size_t>(tensor.offset()),
        .nbytes = tensor.nbytes(),
        .rank = tensor.rank()
    };
}


} // namespace tannic 