#include "shape.h"
#include <iostream>
#include <stdio.h>
#include "CudaException.h"

using namespace std;

int main(void)
{
    Shape shape = Shape(100, 200);
    cout << "shape x: " << shape.x << ", shape y: " << shape.y << endl;

    float* d_data;
    cudaError_t error = cudaMalloc(&d_data, 100*sizeof(float));
    CudaException::throw_if_error("cudaMalloc");

    error = cudaFree(d_data);
    CudaException::throw_if_error("cudaFree");
    return 0;
}