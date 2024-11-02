#include <iostream>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iomanip>
#include <float.h>

using namespace std;

void generate_points(float3* points, int N){
        for (int i = 0; i < N; i++){
                points[i].x = rand() / (float)RAND_MAX;
                points[i].y = rand() / (float)RAND_MAX;
                points[i].z = rand() / (float)RAND_MAX;
        }
}
void print_points(float3* points, int N){
        for (int i = 0; i < N; i++){
                cout << "Point " << i << ": (" << points[i].x << ", " << points[i].y << ", " << points[i].z << ")" << endl;
        }
}
void print_result(int* results, int N){
        for (int i = 0; i < N; i++){
                cout << "Point " << i << " is closest to point " << results[i] << endl;
        }
}

__global__ void find_closest_points(float3* points, int N, int *results, int threads_per_block){
        int thread = threadIdx.x;
        int block = blockIdx.x;
        int i = block * threads_per_block + thread;

        if (i >= N){
                return;
        }

        // printf("Hello from i: thread: %d block: %d i: %d\n", thread, block, i );

        float3 p = points[i];
        float distance_to_closest = FLT_MAX;

        for (int j = 0; j < N; j++){
                if (i == j) continue;
                float3 q = points[j];

                float dist_sqrd = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z);

                if (dist_sqrd < distance_to_closest){
                        distance_to_closest = dist_sqrd;
                        results[i] = j;
                }
        }
}

int main(){
        srand(1);

        int N = 100000;
        float3* points = new float3[N];

        generate_points(points, N);
        clock_t start = clock();
        float3 *d_points;
        int  *d_results;

        cudaMalloc(&d_points, sizeof(float3)*N);
        cudaMalloc(&d_results, sizeof(int)*N);


        cudaMemcpy(d_points, points, N * sizeof(float3), cudaMemcpyHostToDevice);

        int number_of_threads_per_block = 1024;
        int number_of_blocks = (N / number_of_threads_per_block) + 1;

        find_closest_points<<<number_of_blocks, number_of_threads_per_block>>>(d_points, N, d_results, number_of_threads_per_block);
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        int* results = new int[N];
        cudaMemcpy(results, d_results, N *sizeof(int), cudaMemcpyDeviceToHost);

        // print_result(results, N);

        clock_t end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);

        cout << "Time taken by program is : " << fixed << setprecision(5) << time_taken << " sec " << endl;

        delete[] points;
        delete[] results;
        return 0;
}