#include "layer.h"

class LinearLayer : public Layer {
public:
    LinearLayer(std::string name, uint32_t input_size, uint32_t output_size, float init_scale = 1.0);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& input, const Matrix& existing_grad) override;
    void update_parameters(float learning_rate);
    void initialize_weights();
    void clip_gradients(float min_val, float max_val);

    Matrix weights;
    Matrix weights_grad;

    Matrix biases;
    Matrix biases_grad;
};
