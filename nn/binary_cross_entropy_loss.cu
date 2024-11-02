#include <assert.h>
#include "binary_cross_entropy_loss.h"
#include "cuda_exception.h"

__global__ void forward_kernel(const float* y_true, const float* y_pred, float *loss, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size){
        return;
    }
    float pred_value = y_pred[idx];
    // we dont want to do log of zero, or do a log of 1 and get zero, so we will clamp to 0 to 1 range, 
    pred_value = fmaxf(fminf(pred_value, 1.0f - 1e-7), 1e-7);

    float true_value = y_true[idx];


    float cost_for_this_idx = -(true_value * logf(pred_value) + (1.0f-true_value)*logf(1.0f-pred_value))/size;

    atomicAdd(loss, cost_for_this_idx);
}

__global__ void backward_kernel(const float* y_true, const float* y_pred, float *grad, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size){
        return;
    }

    float pred_value = y_pred[idx];
    pred_value = fmaxf(fminf(pred_value, 1.0f - 1e-7), 1e-7);

    float true_value = y_true[idx];
    grad[idx] = ((-true_value / pred_value) + (1-true_value)/(1-pred_value));
}

float BinaryCrossEntropyLoss::forward(const Matrix& y_true, const Matrix& y_pred) {
    if (y_true.shape().x != y_pred.shape().x) {
        throw std::runtime_error("Shape mismatch in BinaryCrossEntropyLoss forward, got " + std::to_string(y_true.shape().x) + " and " + std::to_string(y_pred.shape().x));
    }

    if (y_true.shape().y != 1 || y_pred.shape().y != 1){
        throw std::runtime_error("number of columns must be 1 for the y vector!, got " + std::to_string(y_true.shape().y) + " and " + std::to_string(y_pred.shape().y));
    }

    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    dim3 block_size = 256;
    dim3 number_of_blocks = ((y_pred.shape().x + (block_size.x - 1)) / block_size.x);
    forward_kernel<<<number_of_blocks, block_size>>>(y_true.gpu_data_ptr.get(), y_pred.gpu_data_ptr.get(), d_loss, y_true.shape().x);
    cudaDeviceSynchronize();
    CudaException::throw_if_error("Failed in binary cross entropy forward");

    float loss;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return loss;
}

// Will return a matrix of shape (n, 1), where n is y_true.shape().x
// The matrix will have copied its data to the CPU already
Matrix BinaryCrossEntropyLoss::backward(const Matrix& y_true, const Matrix& y_pred) {
    Matrix grad(y_true.shape());
    if (y_true.shape().x != y_pred.shape().x) {
        throw std::runtime_error("Shape mismatch in BinaryCrossEntropyLoss forward, got " + std::to_string(y_true.shape().x) + " and " + std::to_string(y_pred.shape().x));
    }
    if (y_true.shape().y != 1 || y_pred.shape().y != 1){
        throw std::runtime_error("number of columns must be 1 for the y vector!, got " + std::to_string(y_true.shape().y) + " and " + std::to_string(y_pred.shape().y));
    }
    dim3 block_size = 256;
    dim3 number_of_blocks = ((y_pred.shape().x + (block_size.x - 1)) / block_size.x);
    backward_kernel<<<number_of_blocks, block_size>>>(y_true.gpu_data_ptr.get(), y_pred.gpu_data_ptr.get(), grad.gpu_data_ptr.get(), y_true.shape().x);

    grad.copy_to_cpu();
    return grad;
}

