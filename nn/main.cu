#include "shape.h"
#include <iostream>
#include <stdio.h>
#include "matrix.h"
#include "binary_cross_entropy_loss.h"
using namespace std;

void test_matrix() {
    Matrix matrix = Matrix(5, 3);
    cout << "on cpu at start: \n" << matrix << endl;
    int counter = 0;
    for (size_t i = 0; i < matrix.shape().x; ++i) {
        for (size_t j = 0; j < matrix.shape().y; ++j) {
            matrix(i, j) = counter++;
        }
    }
    cout << "on cpu after init: \n" << matrix << endl;

    // transfer to gpu
    matrix.copy_to_gpu();
    for (size_t i = 0; i < matrix.shape().x; ++i) {
        for (size_t j = 0; j < matrix.shape().y; ++j) {
            matrix(i, j) = 0;
        }
        cout << endl;
    }
    cout << "on cpu after clear: \n" << matrix << endl;

    // transfer to cpu
    matrix.copy_to_cpu();
    cout << "on cpu after back cpu: \n" << matrix << endl;

}


int main(void)
{
    Shape shape = Shape(100, 200);
    cout << "shape x: " << shape.x << ", shape y: " << shape.y << endl;

    test_matrix();
    return 0;
}