#pragma once
#include "matrix.h"

class BinaryCrossEntropyLoss{
public:
    BinaryCrossEntropyLoss(const std::string& name);
    float forward(const Matrix& y_true, const Matrix& y_pred);
    Matrix backward(const Matrix& y_true, const Matrix& y_pred);
    const std::string name;
};