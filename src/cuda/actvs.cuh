#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
#include <tannic/runtime/types.h>
#include <tannic/runtime/tensor.h>
#include <tannic/runtime/status.h>
#include <tannic/runtime/streams.h>

using namespace tannic;

namespace cuda::nn {

status relu(const tensor_t*, tensor_t*, stream_t);
status silu(const tensor_t*, tensor_t*, stream_t); 
status gelu(const tensor_t*, tensor_t*, stream_t);

}