
#include "layer.h"

class LinearLayer : public Layer {
public:
    LinearLayer(std::string name, uint32_t input_size, uint32_t output_size);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& input, const Matrix& grad_output) override;

private:
    Matrix weights_;
    Matrix biases_;
};