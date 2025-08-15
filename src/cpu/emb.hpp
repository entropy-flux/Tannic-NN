#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <tannic/runtime/types.h>
#include <tannic/runtime/tensor.h>
#include <tannic/runtime/status.h>

using namespace tannic;

namespace cpu::nn {

status embed(const tensor_t*, const tensor_t*, tensor_t*);
status embed(const tensor_t*, const tensor_t*, tensor_t*); 

}