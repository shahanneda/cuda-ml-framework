#include <memory>
#include <iostream>
#include <cuda.h>
#include <cstring>

#include "cuda_exception.h"
#include "matrix.h"

using std::cout;
using std::endl;
Matrix::Matrix(Shape shape) : shape_(shape), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {
    allocate_memory();
}

Matrix::Matrix(size_t rows, size_t cols) : shape_(rows, cols), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {
    allocate_memory();
}

Matrix::~Matrix() {
    cudaDeviceSynchronize();
    if (gpu_data_ptr){
        gpu_data_ptr.reset();
        CudaException::throw_if_error("Failed to reset GPU data pointer");
    }
    if (cpu_data_ptr){
        cpu_data_ptr.reset();
    }
}

void Matrix::allocate_memory() {
    allocate_cpu_memory();
    allocate_gpu_memory();
}

Shape Matrix::shape() const {
    return shape_;
}


void Matrix::allocate_cpu_memory() {
    cpu_data_ptr = std::shared_ptr<float>(new float[shape_.x * shape_.y], [](float* p) { delete[] p; });
}

void Matrix::allocate_gpu_memory() {
    float* gpu_data = nullptr;
    size_t bytes_to_allocate = std::max(shape_.x * shape_.y, (size_t)1) * sizeof(float);
    cudaError_t err = cudaMalloc(&gpu_data, bytes_to_allocate);
    if (err != cudaSuccess) {
        throw CudaException("Failed to allocate GPU memory");
    }
    gpu_data_ptr = std::shared_ptr<float>(gpu_data, [](float* p) {cudaFree(p);});
}

float& Matrix::operator()(size_t row, size_t col) {
    int index = row * shape_.y + col;
    has_propagated_updates_to_gpu = false;
    return cpu_data_ptr.get()[index];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    int index = row * shape_.y + col;
    return cpu_data_ptr.get()[index];
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    // matrix.copy_to_cpu();
    os << "[";
    for (size_t i = 0; i < matrix.shape_.x; ++i) {
        for (size_t j = 0; j < matrix.shape_.y; ++j) {
            char last = (static_cast<size_t>(i) == matrix.shape_.x - 1 && static_cast<size_t>(j) == matrix.shape_.y - 1) ? ']' : ' ';
            char first = (static_cast<size_t>(i) == 0 && static_cast<size_t>(j) == 0) ? '\0' : ' ';
            os << first << matrix(i, j) << last;
        }
        os << std::endl;
    }
    return os;
}

int Matrix::total_size() const {
    return shape_.x * shape_.y;
}

void Matrix::copy_to_gpu() const{
    cudaMemcpy(gpu_data_ptr.get(), cpu_data_ptr.get(), total_size() * sizeof(float),  cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    has_propagated_updates_to_gpu = true;
    CudaException::throw_if_error("Failed to copy to GPU");
}

void Matrix::copy_to_cpu() const {
    cudaMemcpy(cpu_data_ptr.get(), gpu_data_ptr.get(), total_size() * sizeof(float),  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CudaException::throw_if_error("Failed to copy to CPU");
}

float& Matrix::operator[](size_t index) {
    return cpu_data_ptr.get()[index];
}

const float& Matrix::operator[](size_t index) const {
    return cpu_data_ptr.get()[index];
}

void Matrix::setIdentity(){
    has_propagated_updates_to_gpu = false;
    for (size_t i = 0; i < shape_.x; ++i){
        for (size_t j = 0; j < shape_.y; ++j){
            (*this)(i, j) = (i == j) ? 1 : 0;
        }
    }
}

void Matrix::set_row(size_t row, const std::vector<float>& values) {
    has_propagated_updates_to_gpu = false;
    if (values.size() != shape_.y) {
        throw std::runtime_error("Invalid number of values to set, expected " + std::to_string(shape_.y) + " got " + std::to_string(values.size()));
    }

    for (size_t i = 0; i < shape_.y; ++i) {
        cpu_data_ptr.get()[row * shape_.y + i] = values[i];
    }
}

void Matrix::set_col(size_t col, const std::vector<float>& values) {
    has_propagated_updates_to_gpu = false;
    if (values.size() != shape_.x) {
        throw std::runtime_error("Invalid number of values to set, expected " + std::to_string(shape_.x) + " got " + std::to_string(values.size()));
    }

    for (size_t i = 0; i < shape_.x; ++i) {
        cpu_data_ptr.get()[i * shape_.y + col] = values[i];
    }
}

__global__ void transpose_kernel(float* M, float* out, uint32_t input_rows, uint32_t input_cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;

    if (row >= input_rows || col >= input_cols){
        return;
    }
    out[col*input_rows + row] = M[row*input_cols + col];
}

Matrix Matrix::T() const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(shape_.y, shape_.x);
    copy_to_gpu();
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    transpose_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols());
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;
    out.copy_to_cpu();
    return out;
}

__global__ void multiply_kernel(float* A, float* B, float* out, int a_rows, int a_cols, int b_rows, int b_cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int out_rows = a_rows;
    int out_cols = b_cols;
    if (a_cols != b_rows){
        // throw error
        printf("Invalid matrix dimensions for multiplication, A cols: %d, B rows: %d\n", a_cols, b_rows);
        return;
    }

    if (row >= out_rows || col >= out_cols){
        return;
    }

    float sum  = 0;
    for (int i = 0; i < a_cols; i++){
        // sum += a[row][i] * b[i][col]
        sum += A[row*a_cols + i] * B[i*b_cols + col];
    }

    // out[row][col] = sum;
    out[row*out_cols + col] = sum;
}

Matrix Matrix::operator*(const Matrix& other) const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    if (!other.has_propagated_updates_to_gpu){
        throw std::runtime_error("Other matrix has not been propagated to GPU");
    }
    // If the multiplication is A*B
    // This is A
    // other is B
    if (cols() != other.rows()){
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix out = Matrix(rows(), other.cols());
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (other.cols() + threads_per_block.y - 1) / threads_per_block.y);

    multiply_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), other.gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols(), other.rows(), other.cols());
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;
    out.copy_to_cpu();
    return out;
}

__global__ void multiply_scalar_kernel(float* M, float* out, int rows, int cols, float scalar){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols){
        return;
    }
    out[row*cols + col] = M[row*cols + col] * scalar;
}

Matrix Matrix::operator*(float scalar) const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(rows(), cols());
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    multiply_scalar_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols(), scalar);
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;

    out.copy_to_cpu();
    return out;
}


__global__ void add_kernel(float* A, float* B, float* out, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols){
        return;
    }
    out[row*cols + col] = A[row*cols + col] + B[row*cols + col];
}

Matrix Matrix::operator+(const Matrix& other) const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    if (!other.has_propagated_updates_to_gpu){
        throw std::runtime_error("Other matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(rows(), cols());
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    add_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), other.gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols());
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;
    out.copy_to_cpu();
    return out;
}

Matrix Matrix::operator-(const Matrix& other) const{
    return *this + (other * -1);
}

__global__ void sum_rows_kernel(float* M, float* out, int rows, int cols){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows){
        return;
    }
    float sum = 0;
    for (int i = 0; i < cols; i++){
        sum += M[row*cols + i];
    }
    out[row] = sum;
}

Matrix Matrix::sum_rows() const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(rows(), 1);
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    sum_rows_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols());
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;
    out.copy_to_cpu();
    return out;
}


__global__ void sum_cols_kernel(float* M, float* out, int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols){
        return;
    }
    float sum = 0;
    for (int i = 0; i < rows; i++){
        sum += M[i*cols + col];
    }
    out[col] = sum;
}

Matrix Matrix::sum_cols() const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(1, cols());
    dim3 threads_per_block(16, 16, 1);  
    dim3 number_of_blocks((cols() + threads_per_block.x - 1) / threads_per_block.x, (rows() + threads_per_block.y - 1) / threads_per_block.y);

    sum_cols_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols());
    cudaDeviceSynchronize();
    out.copy_to_cpu();
    out.set_data_in_gpu_as_valid();
    return out;
}

__global__ void clip_kernel(float* M, float* out, int rows, int cols, float min_val, float max_val){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols){
        return;
    }
    out[row*cols + col] = fminf(fmaxf(M[row*cols + col], min_val), max_val);
}

Matrix Matrix::clip(float min_val, float max_val) const{
    if (!has_propagated_updates_to_gpu){
        throw std::runtime_error("Matrix has not been propagated to GPU");
    }
    Matrix out = Matrix(rows(), cols());
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    clip_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols(), min_val, max_val);
    cudaDeviceSynchronize();
    out.has_propagated_updates_to_gpu = true;
    out.copy_to_cpu();
    return out;
}

// Marks the data currently in the GPU memoery as valid
// So no need to copy data from the CPU to the GPU if this is set
void Matrix::set_data_in_gpu_as_valid(){
    has_propagated_updates_to_gpu = true;
}

// Copy constructor

Matrix::Matrix(const Matrix& other) 
    : shape_(other.shape_), 
      has_propagated_updates_to_gpu(other.has_propagated_updates_to_gpu) {
    allocate_memory();
    // Copy CPU data
    std::memcpy(cpu_data_ptr.get(), other.cpu_data_ptr.get(), total_size() * sizeof(float));
    // Copy GPU data if valid
    if (!other.has_propagated_updates_to_gpu) {
        throw std::runtime_error("Other matrix has not been propagated to GPU");
    }
    cudaMemcpy(gpu_data_ptr.get(), other.gpu_data_ptr.get(), 
                total_size() * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        // Reallocate if shapes don't match
        if (shape_.x != other.shape_.x || shape_.y != other.shape_.y) {
            throw std::runtime_error("Invalid dimensions for assignment. this shape is " + std::to_string(shape_.x) + "x" + std::to_string(shape_.y) + " and other shape is " + std::to_string(other.shape_.x) + "x" + std::to_string(other.shape_.y));
        }
        
        has_propagated_updates_to_gpu = other.has_propagated_updates_to_gpu;
        // Copy CPU data
        std::memcpy(cpu_data_ptr.get(), other.cpu_data_ptr.get(), 
                   total_size() * sizeof(float));
        // Copy GPU data if valid
        if (!other.has_propagated_updates_to_gpu) {
            throw std::runtime_error("Other matrix has not been propagated to GPU");
        }
        cudaMemcpy(gpu_data_ptr.get(), other.gpu_data_ptr.get(), 
                  total_size() * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return *this;
}

bool Matrix::gpu_data_is_valid() const{
    return has_propagated_updates_to_gpu;
}