#include "ops.h"
#include <cmath>

Tensor linear(const Tensor& A,const Tensor& B)
{
    Tensor C(A.rows,B.cols);

    for(int i=0;i<A.rows;i++)
        for(int j=0;j<B.cols;j++)
        {
            float sum=0;

            for(int k=0;k<A.cols;k++)
                sum+=A.data[i*A.cols+k]
                    *B.data[k*B.cols+j];

            C(i,j)=sum;
        }

    return C;
}

Tensor relu(const Tensor& A)
{
    Tensor B(A.rows,A.cols);

    for(int i=0;i<A.rows*A.cols;i++)
        B.data[i]=std::max(0.0f,A.data[i]);

    return B;
}