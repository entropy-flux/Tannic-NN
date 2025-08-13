#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include "Parameters.hpp" 
 
using namespace tannic;

constexpr nn::Parameter X(float32, {4, 4});

TEST(TestParameters, TestParameters) {      
    nn::Parameters parameters; parameters.initialize(4*4*dsizeof(float32)); 
    X.initialize("my-weight", parameters);
    X[0,0] = 1;
    X[0,1] = 2;
    X[1,0] = 3;
    X[1,1] = 4;
    ASSERT_EQ((X[0,0] == 1), true); 
    ASSERT_EQ((X[0,1] == 2), true); 
    ASSERT_EQ((X[1,0] == 3), true); 
    ASSERT_EQ((X[1,1] == 4), true); 
}