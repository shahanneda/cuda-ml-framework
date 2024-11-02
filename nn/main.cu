#include "shape.h"
#include <iostream>
#include <stdio.h>
#include "CudaException.h"
#include "matrix.h"
using namespace std;

int main(void)
{
    Shape shape = Shape(100, 200);
    cout << "shape x: " << shape.x << ", shape y: " << shape.y << endl;

    float* d_data;
    cudaError_t error = cudaMalloc(&d_data, 100*sizeof(float));
    CudaException::throw_if_error("cudaMalloc");

    Matrix matrix = Matrix(5, 3);
    cout << matrix << endl;
    matrix(0, 0) = 1.0;
    matrix(1, 0) = 1.0;
    matrix(2, 0) = 1.0;
    matrix(3, 0) = 1.0;
    matrix(4, 0) = 1.0;

    matrix.copy_to_gpu();
    matrix.copy_to_cpu();
    CudaException::throw_if_error("copy to gpu and back");
    cout << matrix << endl;

    error = cudaFree(d_data);
    CudaException::throw_if_error("cudaFree");
    return 0;
}