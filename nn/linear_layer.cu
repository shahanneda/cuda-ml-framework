#include "linear_layer.h"
#include "cuda_exception.h"

LinearLayer::LinearLayer(std::string name, uint32_t input_size, uint32_t output_size) : Layer(name), weights(input_size, output_size), biases(1, output_size) {

    // input_size is the number of "columns" in the input matrix
    // output_size is the number of "columns" in the output matrix, which corresponds to the number of neurons in the layer
    // For each row of the output, that represents the value of a single data point, for all the neuroons.
    // so for example, if we [x_1, x_2] ad we multiply it by our weights, xW = n, we will get [n_1, n_2], where n_1 represents the output value of neuron 1 and n_2 represents the output value of neruon 2.
    initialize_weights();
}

__global__ void forward_kernel(float* input, float* weights, float* biases, float* output, int batch_size, int number_of_inputs, int number_of_neurons) {
    int out_row = blockIdx.x * blockDim.x + threadIdx.x; // corresponds to which input vector
    int out_col = blockIdx.y * blockDim.y + threadIdx.y; // corresponds to which neuron
    if (out_row >= batch_size || out_col >= number_of_neurons){
        return;
    }
    float sum = 0; 
    for (int i = 0; i < number_of_inputs; i++){
        // input[out_row][i] * weights[i][out_col]
        sum += input[out_row*number_of_inputs + i] * weights[i*number_of_neurons + out_col];
    }
    sum += biases[out_col];
    output[out_row * number_of_neurons + out_col] = sum;
}

void LinearLayer::initialize_weights() {
    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            weights(i, j) = fmaxf(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), 1e-7f);
        }
    }
    for (int i = 0; i < biases.cols(); i++) {
        biases(0, i) = fmaxf(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), 1e-7f);
    }
}

// make sure input has been copied to the GPU first!
// Will return Matrix of shape (batch_size, number_of_neurons)
Matrix LinearLayer::forward(const Matrix& input) {
    uint32_t batch_size = input.rows();

    uint32_t number_of_inputs = input.cols();
    uint32_t expected_input_size = weights.rows();

    if (number_of_inputs != expected_input_size) {
        throw std::invalid_argument("Input size (" + std::to_string(number_of_inputs) + ") does not match expected input size of " + std::to_string(expected_input_size));
    }

    uint32_t number_of_neurons = weights.cols();

    dim3 number_of_threads_per_pool = dim3(16, 16, 1);
    dim3 number_of_pools = dim3(
        (batch_size + number_of_threads_per_pool.x - 1) / number_of_threads_per_pool.x,
        (number_of_neurons + number_of_threads_per_pool.y - 1) / number_of_threads_per_pool.y,
        1
    );

    Matrix output(batch_size, number_of_neurons);
    weights.copy_to_gpu();
    biases.copy_to_gpu();

    forward_kernel<<<number_of_pools, number_of_threads_per_pool>>>(input.gpu_data_ptr.get(), weights.gpu_data_ptr.get(), biases.gpu_data_ptr.get(), output.gpu_data_ptr.get(), batch_size, number_of_inputs, number_of_neurons);

    CudaException::throw_if_error("Failed to run forward kernel");

    return output;
}

Matrix LinearLayer::backward(const Matrix& input, const Matrix& grad_output) {
    // return grad_output * weights.T();
    return Matrix(0, 0);
}