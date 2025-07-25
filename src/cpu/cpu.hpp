#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <tannic/runtime/types.h>
#include <tannic/runtime/tensor.h>

using namespace tannic;

namespace cpu {

void relu(const tensor_t*, tensor_t*);
void silu(const tensor_t*, tensor_t*); 

}