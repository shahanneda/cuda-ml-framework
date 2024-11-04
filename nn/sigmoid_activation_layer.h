#pragma once

#include "layer.h"
#include "matrix.h"

class SigmoidActivationLayer : public Layer {
public:
    SigmoidActivationLayer(std::string name);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& input, const Matrix& grad_output) override;
};