#include <iostream>
#include <tannic.hpp>
#include <tannic-nn.hpp>
#include <tannic-nn/functional.hpp>

using namespace tannic;

int main() { 
    Tensor X(float32, {2, 3, 4});
    X.initialize({
        { 
            {0.1, -0.2, 0.3, -0.4},
            {0.5, -0.6, 0.7, -0.8},
            {0.9, -1.0, 1.1, -1.2}
        },
        { 
            {1.3, -1.4, 1.5, -1.6},
            {1.7, -1.8, 1.9, -2.0},
            {2.1, -2.2, 2.3, -2.4}
        }
    });

    Tensor Y = nn::gelu(X.transpose(0, 1)); 

    Tensor Y_expected(float32, {3, 2, 4});
    Y_expected.initialize({
        {
            { 0.0540, -0.0841, 0.1854, -0.1378 }, 
            { 1.1742, -0.1131, 1.3998, -0.0877 }
        },
        { 
            { 0.3457, -0.1646, 0.5306, -0.1695 },
            { 1.6242, -0.0647, 1.8454, -0.0455 }
        },
        {
            { 0.7343, -0.1587, 0.9508, -0.1381 },
            { 2.0625, -0.0306, 2.2753, -0.0197 }
        }
    });  
 

    Tensor tensor1(float32, {1, 3, 4});
    tensor1.initialize({
        {
            { -0.3557, -0.1649, -1.1071, 0.2518 },
            { -0.2209, -0.2393, -1.3707, 0.2814 },
            { -0.3827, -0.2535, -1.2362, 0.4319 }
        }
    });
 
    Tensor tensor2_expected(float32, {1, 3, 4});
    tensor2_expected.initialize({
        {
            { -0.1284, -0.0717, -0.1485, 0.1509 },
            { -0.0911, -0.0970, -0.1168, 0.1718 },
            { -0.1343, -0.1014, -0.1337, 0.2881 }
        }
    });
    
    Tensor tesnor2 = nn::gelu(tensor1);

    std::cout << tesnor2 << tensor2_expected;
}