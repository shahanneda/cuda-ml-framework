#include "Matrix.h"

Matrix::Matrix(Shape shape) : shape_(shape), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {

}
Matrix::Matrix(size_t rows, size_t cols) : shape_(rows, cols), cpu_data_ptr(nullptr), gpu_data_ptr(nullptr) {
}

void Matrix::allocate_memory() {
    allocate_cpu_memory();
    allocate_gpu_memory();
}