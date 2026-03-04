#pragma once
#include "tensor.h"

Tensor linear(
    const Tensor& input,
    const Tensor& weight
);

Tensor relu(
    const Tensor& input
);

Tensor softmax(
    const Tensor& input
);