#include "linear_layer.h"
#include "cuda_exception.h"
#include <iostream>
using namespace std;

LinearLayer::LinearLayer(std::string name, uint32_t input_size, uint32_t output_size, float init_scale) : Layer(name), weights(input_size, output_size), biases(1, output_size), weights_grad(input_size, output_size), biases_grad(1, output_size) {

    // input_size is the number of "columns" in the input matrix
    // output_size is the number of "columns" in the output matrix, which corresponds to the number of neurons in the layer
    // For each row of the output, that represents the value of a single data point, for all the neuroons.
    // so for example, if we [x_1, x_2] ad we multiply it by our weights, xW = n, we will get [n_1, n_2], where n_1 represents the output value of neuron 1 and n_2 represents the output value of neruon 2.
    initialize_weights();
    weights.copy_to_gpu();
    biases.copy_to_gpu();
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


    // // Initialize weights with scaled random values
    // for (size_t i = 0; i < weights.shape().x; ++i) {
    //     for (size_t j = 0; j < weights.shape().y; ++j) {
    //         // Xavier/Glorot initialization scaled by init_scale
    //         float limit = sqrt(6.0f / (input_size + output_size));
    //         weights(i, j) = init_scale * ((float)rand() / RAND_MAX * 2.0f - 1.0f) * limit;
    //     }
    // }
    
    // // Initialize biases to zero
    // for (size_t i = 0; i < biases.shape().x; ++i) {
    //     biases(i, 0) = 0.0f;
    // }

    for (int i = 0; i < weights.rows(); i++) {
        for (int j = 0; j < weights.cols(); j++) {
            weights(i, j) = fmaxf(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), 1e-7f);
        }
    }
    for (int i = 0; i < biases.cols(); i++) {
        // biases(0, i) = fmaxf(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), 1e-7f);
        biases(i, 0) = 0.0f;
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
    cudaDeviceSynchronize();
    output.set_data_in_gpu_as_valid();

    CudaException::throw_if_error("Failed to run forward kernel");
    output.copy_to_cpu();

    return output;
}

Matrix LinearLayer::backward(const Matrix& input, const Matrix& grad_output) {
    // this is the deritivate of the loss, with respect to the weights
    // we know dL/dW = dL/dZ * dZ /dW, where Z = XW + b
    // dL/dZ is grad_output
    // dZ/dW is the input

    // grad_output is of shape (batch_size, outputs)
    // input is of shape (batch_size, inputs)
    // weights_grad is of shape (inputs, outputs)
    // biases_grad is of shape (1, outputs)

    if (input.rows() != grad_output.rows()){
        throw std::invalid_argument("Input and grad_output must have the same number of rows");
    }
    if (input.cols() != weights.rows()){
        throw std::invalid_argument("Input and weights must have the same number of columns");
    }

    input.copy_to_gpu();
    grad_output.copy_to_gpu();
    
    // compute gradients for parameters
    weights_grad = input.T() * grad_output;
    // cout << "grad output.shape(): " << grad_output.shape().x << "x" << grad_output.shape().y << endl;
    biases_grad = grad_output;
    cout << "biases_grad: \n" << biases_grad << endl;
    cout << "weights_grad: \n" << weights_grad << endl;
    
    // compute gradient to propagate backward: dL/dx = dL/dy * W^T
    weights.copy_to_gpu();  // Ensure weights are on GPU
    return grad_output * weights.T();
}

void LinearLayer::update_parameters(float learning_rate){
    cout << "weights_grad: \n" << weights_grad << endl;
    cout << " before updating weights: \n" << weights << endl;
    weights = weights - (weights_grad * learning_rate);
    weights.copy_to_cpu();
    // cout << " after updating weights: \n" << weights << endl;

    // cout << "biases_grad: \n" << biases_grad << endl;
    // cout << " before updating biases: \n" << biases << endl;
    // biases_grad.copy_to_cpu();
    // cout << "biases_grad: \n" << biases_grad << endl;
    // cout << "biases grad * learning rate: \n" << biases_grad * learning_rate << endl;
    // biases_grad.copy_to_cpu();

    // Debug print the memory addresses
    // cout << "weights_grad GPU ptr: " << weights_grad.gpu_data_ptr.get() << endl;
    // cout << "weights GPU ptr: " << weights.gpu_data_ptr.get() << endl;
    // cout << "biases_grad GPU ptr: " << biases_grad.gpu_data_ptr.get() << endl;
    // cout << "biases GPU ptr: " << biases.gpu_data_ptr.get() << endl;

    // Do the weights_grad + weights operation separately
    // Matrix temp = weights_grad + weights;
    // cudaDeviceSynchronize();
    // cout << "random line: \n"
    //      << temp << endl;
    // cudaDeviceSynchronize();

    // cout << "biases: \n" << biases << endl;
    // cout << "biases_grad: \n" << biases_grad << endl;
    // cout << "biases_grad * learning rate: \n" << biases_grad * learning_rate << endl;

    biases = biases - (biases_grad * learning_rate);
    biases.copy_to_cpu();
    cout << " after updating biases: \n" << biases << endl;
}

void LinearLayer::clip_gradients(float min_val, float max_val){
    weights_grad = weights_grad.clip(min_val, max_val);
    biases_grad = biases_grad.clip(min_val, max_val);
}