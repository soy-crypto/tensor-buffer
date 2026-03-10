#pragma once
#include <vector>
#include "ops.h"

class Graph 
{
    private:
        std::vector<Operator*> ops;

    public:
        void add_op(Operator* op);
        Tensor run(const Tensor& input);
};