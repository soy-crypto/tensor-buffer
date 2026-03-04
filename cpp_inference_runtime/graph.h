#pragma once

#include "tensor.h"
#include <vector>

class Layer {

public:

    virtual Tensor forward(
        const Tensor& input
    ) = 0;

};