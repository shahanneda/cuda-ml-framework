
#include "layer.h"

class LinearLayer : public Layer {
public:
    LinearLayer(std::string name, uint32_t input_size, uint32_t output_size);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& input, const Matrix& existing_grad) override;
    void initialize_weights();

    Matrix weights;
    Matrix weights_grad;

    Matrix biases;
    Matrix biases_grad;
};