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

void test_binary_cross_entropy_loss() {
    Matrix y_true = Matrix(10, 1);
    Matrix y_pred = Matrix(10, 1);
    y_true.set_col(0, {0, 0, 1, 0, 1, 0, 1, 1, 1, 0});
    y_pred.set_col(0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    cout << "y_pred: \n" << y_pred << endl;
    cout << "y_true: \n" << y_true << endl;
    y_true.copy_to_gpu();
    y_pred.copy_to_gpu();
    BinaryCrossEntropyLoss loss;
    float loss_value = loss.forward(y_true, y_pred);

    cout << "loss value: " << loss_value << endl;

    Matrix loss_grad = loss.backward(y_true, y_pred);
    loss_grad.copy_to_cpu();
    cout << "loss grad: \n" << loss_grad << endl;
}

int main(void)
{
    test_binary_cross_entropy_loss();
    return 0;
}