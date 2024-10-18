#include <iostream>
#include <cuda.h>


using namespace std;

__global__ void add(int *a, int *b){
        a[0] = a[0] = b[0];
}

int main() {
    int a = 89;
    int b = 49;
    cout << "a: " << a << " b: " << b << endl;

    int *d_a, *d_b;

    // allocate some memory on the device
    cudaMalloc(&d_a, 2*sizeof(int));
    d_b = d_a + 1;


    // copy the data from the host to the device
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, 1>>>(d_a, d_b);
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "result: a + b = " << a << endl;

    cudaFree(d_a);
    cudaFree(d_b);

    cout << "Hello World!" << endl;
    return 0;
}