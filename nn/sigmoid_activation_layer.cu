#include "sigmoid_activation_layer.h"

SigmoidActivationLayer::SigmoidActivationLayer(std::string name) : Layer(name) {}

__device__ float sigmoid(float x) {
    return 1/(1+exp(-x));
}

__global__ void sigmoid_forward(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = sigmoid(input[idx]);
    }
}

Matrix SigmoidActivationLayer::forward(const Matrix& input) {
    Matrix output(input.shape());
    int number_of_threads = 1024;
    int number_of_elements = input.rows()*input.cols();
    int number_of_blocks = (number_of_elements + number_of_threads - 1) / number_of_threads;

    if(!input.gpu_data_is_valid()){
        throw std::invalid_argument("Input matrix has not been propagated to GPU");
    }

    sigmoid_forward<<<number_of_blocks, number_of_threads>>>(input.gpu_data_ptr.get(), output.gpu_data_ptr.get(), number_of_elements);
    cudaDeviceSynchronize();
    output.set_data_in_gpu_as_valid();
    return output;
}

__global__ void sigmoid_backward(const float* input, const float* grad_output, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = sigmoid(input[idx]);
        output[idx] = grad_output[idx] * sigmoid_val * (1 - sigmoid_val);
    }
}

Matrix SigmoidActivationLayer::backward(const Matrix& input, const Matrix& grad_output) {
    Matrix output(input.shape());
    int number_of_threads = 1024;
    int number_of_elements = input.rows()*input.cols();
    int number_of_blocks = (number_of_elements + number_of_threads - 1) / number_of_threads;

    if (grad_output.rows() != input.rows() || grad_output.cols() != input.cols()) {
        throw std::invalid_argument("grad_output and input must have the same shape");
    }

    if(!grad_output.gpu_data_is_valid()){
        throw std::invalid_argument("grad_output matrix has not been propagated to GPU");
    }

    sigmoid_backward<<<number_of_blocks, number_of_threads>>>(input.gpu_data_ptr.get(), grad_output.gpu_data_ptr.get(), output.gpu_data_ptr.get(), number_of_elements);
    cudaDeviceSynchronize();
    output.set_data_in_gpu_as_valid();
    return output;
}