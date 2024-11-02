#include "matrix.h"
#include <memory>
#include <iostream>
#include <cuda.h>
#include "cuda_exception.h"

Matrix::Matrix(Shape shape) : shape_(shape), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {
    allocate_memory();
}

Matrix::Matrix(size_t rows, size_t cols) : shape_(rows, cols), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {
    allocate_memory();
}

Matrix::~Matrix() {
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
    cudaMalloc(&gpu_data, shape_.x * shape_.y * sizeof(float));
    gpu_data_ptr = std::shared_ptr<float>(gpu_data, [](float* p) {cudaFree(p);});
}

float& Matrix::operator()(size_t row, size_t col) {
    int index = row * shape_.y + col;
    return cpu_data_ptr.get()[index];
}

const float& Matrix::operator()(size_t row, size_t col) const {
    int index = row * shape_.y + col;
    return cpu_data_ptr.get()[index];
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
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

void Matrix::copy_to_gpu() {
    cudaMemcpy(gpu_data_ptr.get(), cpu_data_ptr.get(), total_size() * sizeof(float),  cudaMemcpyHostToDevice);
    CudaException::throw_if_error("Failed to copy to GPU");
}

void Matrix::copy_to_cpu() {
    cudaMemcpy(cpu_data_ptr.get(), gpu_data_ptr.get(), total_size() * sizeof(float),  cudaMemcpyDeviceToHost);
    CudaException::throw_if_error("Failed to copy to CPU");
}

float& Matrix::operator[](size_t index) {
    return cpu_data_ptr.get()[index];
}

const float& Matrix::operator[](size_t index) const {
    return cpu_data_ptr.get()[index];
}

void Matrix::set_row(size_t row, const std::vector<float>& values) {
    if (values.size() != shape_.y) {
        throw std::runtime_error("Invalid number of values to set, expected " + std::to_string(shape_.y) + " got " + std::to_string(values.size()));
    }

    for (size_t i = 0; i < shape_.y; ++i) {
        cpu_data_ptr.get()[row * shape_.y + i] = values[i];
    }
}

void Matrix::set_col(size_t col, const std::vector<float>& values) {
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

Matrix Matrix::T()  {
    Matrix out = Matrix(shape_.y, shape_.x);
    copy_to_gpu();
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (cols() + threads_per_block.y - 1) / threads_per_block.y);

    transpose_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols());
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
    // This is A
    // B is the other one
    if (cols() != other.rows()){
        throw std::invalid_argument("Invalid matrix dimensions for multiplication");
    }
    Matrix out = Matrix(rows(), other.cols());
    dim3 threads_per_block(16, 16, 1);
    dim3 number_of_blocks((rows() + threads_per_block.x - 1) / threads_per_block.x, (other.cols() + threads_per_block.y - 1) / threads_per_block.y);

    multiply_kernel<<<number_of_blocks, threads_per_block>>>(gpu_data_ptr.get(), other.gpu_data_ptr.get(), out.gpu_data_ptr.get(), rows(), cols(), other.rows(), other.cols());
    out.copy_to_cpu();
    return out;
}
