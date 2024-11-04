#include "relu_activation_layer.h"

ReluActivationLayer::ReluActivationLayer(std::string name) : Layer(name) {}


__global__ void relu_forward(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0, input[idx]);
    }
}

Matrix ReluActivationLayer::forward(const Matrix& input) {
    Matrix output(input.shape());
    int number_of_threads = 1024;
    int number_of_elements = input.rows()*input.cols();
    int number_of_blocks = (number_of_elements + number_of_threads - 1) / number_of_threads;

    relu_forward<<<number_of_blocks, number_of_threads>>>(input.gpu_data_ptr.get(), output.gpu_data_ptr.get(), number_of_elements);
    cudaDeviceSynchronize();
    output.set_data_in_gpu_as_valid();
    return output;
}


__global__ void relu_backward(const float* input, const float* grad_output, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0) {
            output[idx] = grad_output[idx];
        } else {
            output[idx] = 0;
        }
    }
}

Matrix ReluActivationLayer::backward(const Matrix& input, const Matrix& grad_output) {
    Matrix output(input.shape());
    int number_of_threads = 1024;
    int number_of_elements = input.rows()*input.cols();
    int number_of_blocks = (number_of_elements + number_of_threads - 1) / number_of_threads;

    relu_backward<<<number_of_blocks, number_of_threads>>>(input.gpu_data_ptr.get(), grad_output.gpu_data_ptr.get(), output.gpu_data_ptr.get(), number_of_elements);
    cudaDeviceSynchronize();
    output.set_data_in_gpu_as_valid();
    return output;
}