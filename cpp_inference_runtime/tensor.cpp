#include "tensor.h"

Tensor::Tensor(int r,int c)
{
    rows=r;
    cols=c;
    data.resize(r*c);
}

float& Tensor::operator()(int r,int c)
{
    return data[r*cols+c];
}