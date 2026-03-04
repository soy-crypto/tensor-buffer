#include "tensor.h"
#include "ops.h"
#include <iostream>

int main()
{

    Tensor input(1,512);
    Tensor weight(512,512);

    Tensor out = linear(input,weight);

    out = relu(out);

    out = softmax(out);

    std::cout<<"Inference done\n";

}