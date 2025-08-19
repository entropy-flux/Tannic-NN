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

status softmax(const tensor_t*, tensor_t*, stream_t, uint8_t); 

}