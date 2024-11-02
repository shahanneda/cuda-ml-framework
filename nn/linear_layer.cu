#include "linear_layer.h"

LinearLayer::LinearLayer(std::string name, uint32_t input_size, uint32_t output_size) : Layer(name), weights_(input_size, output_size), biases_(1, output_size) {

    // input_size is the number of "columns" in the input matrix
    // output_size is the number of "columns" in the output matrix, which corresponds to the number of neurons in the layer
    // For each row of the output, that represents the value of a single data point, for all the neuroons.
    // so for example, if we [x_1, x_2] ad we multiply it by our weights, xW = n, we will get [n_1, n_2], where n_1 represents the output value of neuron 1 and n_2 represents the output value of neruon 2.
}

__global__ void forward_kernel(float* input, float* weights, float* biases, float* output, int batch_size, int number_of_inputs, int number_of_neurons) {
    int out_row = blockIdx.x * blockDim.x + threadIdx.x; // corresponds to which input vector
    int out_col = blockIdx.y * blockDim.y + threadIdx.y; // corresponds to which output weight
    if (out_row >= batch_size || out_col >= number_of_neurons){
        return;
    }

    float sum = 0; 
    for (int i = 0; i < number_of_inputs; i++){
        // input[out_row][i] * weights[]
    }

}

Matrix LinearLayer::forward(const Matrix& input) {
    uint32_t batch_size = input.rows();

    uint32_t number_of_inputs = input.cols();
    uint32_t expected_input_size = weights_.rows();

    if (number_of_inputs != expected_input_size) {
        throw std::invalid_argument("Input size (" + std::to_string(number_of_inputs) + ") does not match expected input size of " + std::to_string(expected_input_size));
    }

    uint32_t number_of_neurons = weights_.cols();

    // return input * weights_ + biases_;
}