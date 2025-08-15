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

constexpr uint32_t magic = 0x43495245;

#pragma pack(push, 1)  
struct Header {
    uint32_t magic;      
    uint8_t  protocol;
    uint8_t  code;
    uint16_t checksum;
    size_t nbytes; 
};
#pragma pack(pop)