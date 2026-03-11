#pragma once
#include <vector>

class Tensor 
{
private:
    std::vector<float> data;
    int rows;
    int cols;

public:
    Tensor(int r, int c);
    float& operator()(int r, int c);
    float operator()(int r, int c) const;
    const float* getData() const;
    int getRows() const;
    int getCols() const;

};