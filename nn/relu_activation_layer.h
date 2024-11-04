#include "matrix.h"
#include "layer.h"

class ReluActivationLayer : public Layer {
public:
    ReluActivationLayer(std::string name);
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& input, const Matrix& grad_output);
    virtual ~ReluActivationLayer() = default;
};