#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <tannic.hpp>
#include "Parameters.hpp" 
 
using namespace tannic;

constexpr nn::Parameter X(float32, {4, 4});
constexpr nn::Parameter Y(float32, {4, 4});

TEST(TestParameters, TestParameters) {      
    nn::Parameters parameters; parameters.initialize(8*4*dsizeof(float32));     
    X.initialize("my-weight", parameters);
    X[0,0] = 1;
    X[0,1] = 2;
    X[1,0] = 3;
    X[1,1] = 4;
    ASSERT_EQ((X[0,0] == 1), true); 
    ASSERT_EQ((X[0,1] == 2), true); 
    ASSERT_EQ((X[1,0] == 3), true); 
    ASSERT_EQ((X[1,1] == 4), true);    

    Y.initialize("my-other-weight", parameters);
    Y[0,0] = 10;
    Y[0,1] = 20;
    Y[1,0] = 30;
    Y[1,1] = 40;
    ASSERT_EQ((Y[0,0] == 10), true); 
    ASSERT_EQ((Y[0,1] == 20), true); 
    ASSERT_EQ((Y[1,0] == 30), true); 
    ASSERT_EQ((Y[1,1] == 40), true);  
}