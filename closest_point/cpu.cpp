#include <iostream>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iomanip>

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
void print_result(float* results, int N){
        for (int i = 0; i < N; i++){
                cout << "Point " << i << " is closest to point " << results[i] << endl;
        }
}

void find_closest_points(float3* points, int N, float *results){
        for (int i = 0; i < N; i++){
                float3 p = points[i];
                float distance_to_closest = std::numeric_limits<float>::max();

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
}

int main(){
        srand(1);

        int N = 10000;
        float3* points = new float3[N];

        generate_points(points, N);
        // print_points(points, N);

        // time the function
        clock_t start = clock();
        float* result = new float[N];
        find_closest_points(points, N, result);
        // print_result(result, N);
        clock_t end = clock();
        double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
        cout << "Time taken by program is : " << fixed << setprecision(5) << time_taken;
        cout << " sec " << endl;


        delete[] points;
        delete[] result;
        return 0;
}