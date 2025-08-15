#include <gtest/gtest.h>
#include "Parameters.hpp"
#include "Embeddings.hpp"

using namespace tannic;

constexpr nn::Embedding W(float32 ,4, 4);  

TEST(EmbeddingTest, TestEmbeddings) {
    nn::Parameters parameters; parameters.initialize(1024);   
    W.initialize("test_embed", parameters);  
    W[0,0] = 1;
    W[0,1] = 2; 
    W[0,2] = 3; 
    W[0,3] = 4; 
    W[1,0] = 5;
    W[1,1] = 6; 
    W.weight()[1,2] = 7; 
    W.weight()[1,3] = 8;  
    W.weight()[2,0] = 1;
    W.weight()[2,1] = 2; 
    W.weight()[2,2] = 3; 
    W.weight()[2,3] = 4; 
    W.weight()[3,0] = 5;
    W.weight()[3,1] = 6; 
    W.weight()[3,2] = 7; 
    W.weight()[3,3] = 8;  

    Tensor X = W(0, 3);

    ASSERT_EQ(X.shape(), Shape(2,4));
    ASSERT_EQ((X[0,0] == 1),true);
    ASSERT_EQ((X[0,1] == 2),true);
    ASSERT_EQ((X[0,2] == 3),true);
    ASSERT_EQ((X[0,3] == 4),true);
    ASSERT_EQ((X[1,0] == 5),true);
    ASSERT_EQ((X[1,1] == 6),true);
    ASSERT_EQ((X[1,2] == 7),true);
    ASSERT_EQ((X[1,3] == 8),true);

    Tensor I(int64, {2}); I.initialize();
    I[0] = 0;
    I[1] = 3;

    X = W(I);
    ASSERT_EQ((X[0,0] == 1),true);
    ASSERT_EQ((X[0,1] == 2),true);
    ASSERT_EQ((X[0,2] == 3),true);
    ASSERT_EQ((X[0,3] == 4),true);
    ASSERT_EQ((X[1,0] == 5),true);
    ASSERT_EQ((X[1,1] == 6),true);
    ASSERT_EQ((X[1,2] == 7),true);
    ASSERT_EQ((X[1,3] == 8),true); 
}