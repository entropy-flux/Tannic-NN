#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <list>
#include <algorithm> 
#include <tannic.hpp> 

class MNIST {
public:
    MNIST(std::string const& filepath) { 
        std::array<char, 4> header;
        std::array<std::byte, 4> magic =  {std::byte('M'), std::byte('L'), std::byte('B'), std::byte('C') }; 
        std::ifstream file(filepath, std::ios::binary);
        if (file.is_open()) { 
            file.read(header.data(), 4); 
        }
    
        if (!std::equal(header.begin(), header.begin() + 4, magic.begin())) {
            std::cerr << "Invalid magic header\n";
            return;
        }  
    } 
};