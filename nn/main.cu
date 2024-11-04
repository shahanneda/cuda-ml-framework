#include "shape.h"
#include <iostream>
#include <stdio.h>
#include "matrix.h"
#include "binary_cross_entropy_loss.h"
#include "linear_layer.h"
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

void test_transpose(){
    Matrix matrix = Matrix(3, 2);
    matrix.set_row(0, {1, 2});
    matrix.set_row(1, {3, 4});
    matrix.set_row(2, {5, 6});
    cout << "matrix: \n" << matrix << endl;
    Matrix transposed = matrix.T();
    cout << "transposed: \n" << transposed << endl;
}

void test_matrix_multiplication(){
    Matrix a = Matrix(2, 3);
    Matrix b = Matrix(3, 2);
    a.set_row(0, {1, 2, 3});
    a.set_row(1, {4, 5, 6});
    b.set_row(0, {7, 8});
    b.set_row(1, {9, 10});
    b.set_row(2, {11, 12});
    a.copy_to_gpu();
    b.copy_to_gpu();
    cout << "a: \n" << a << endl;
    cout << "b: \n" << b << endl;
    Matrix c = a * b;
    c.copy_to_cpu();
    cout << "c: \n" << c << endl;
}


void test_linear_layer(){
    LinearLayer layer("test_linear_layer", 2, 1);
    cout << "layer: \n" << layer.weights << endl << layer.biases << endl;

    Matrix input(1, 2);
    input.set_row(0, {1, 2});
    input.copy_to_gpu();
    cout << "input: \n" << input << endl;

    Matrix output = layer.forward(input);
    output.copy_to_cpu();
    cout << "output: \n" << output << endl;

    // we need grad_output to be of shape (batch_size, outputs)
    Matrix grad_output(1, 1);
    grad_output(0, 0) = 1;

    Matrix grad = layer.backward(input, grad_output);
    grad.copy_to_cpu();
    cout << "weights_grad: \n" << layer.weights_grad << endl;
    cout << "biases_grad: \n" << layer.biases_grad << endl;

}

void test_sum_rows(){
    Matrix matrix = Matrix(3, 2);
    matrix.set_row(0, {1, 2});
    matrix.set_row(1, {3, 4});
    matrix.set_row(2, {5, 6});
    cout << "matrix: \n" << matrix << endl;
    Matrix sum = matrix.sum_rows();
    cout << "sum: \n" << sum << endl;
}

void test_sum_cols(){
    Matrix matrix = Matrix(3, 2);
    matrix.set_row(0, {1, 2});
    matrix.set_row(1, {3, 4});
    matrix.set_row(2, {5, 6});
    cout << "matrix: \n" << matrix << endl;
    Matrix sum = matrix.sum_cols();
    cout << "sum: \n" << sum << endl;
}

void test_one_layer() {
    // Initialize layers with much smaller initial weights
    LinearLayer layer_1("test_linear_layer_1", 1, 1, 0.01);

    Matrix input(1, 1);
    input.set_row(0, {0.5});
    input.copy_to_gpu();

    Matrix target(1, 1);
    target(0, 0) = 0.5;
    target.copy_to_gpu();

    // Try a range of learning rates
    float learning_rate = 0.1;

    for (size_t i = 0; i < 100; ++i) {
        // cout << " right before forward" << endl;
        // cout << "input shape: " << input.shape().x << " " << input.shape().y << endl;
        Matrix output_1 = layer_1.forward(input);
        Matrix diff = output_1 - target;

        float loss = (diff * diff).sum_rows()(0, 0) / 2.0f;
        // cout << "loss: " << loss << endl;
        cout << "output_1: \n" << output_1 << endl;

        
        Matrix loss_grad = diff;
        cout << "loss_grad: \n" << loss_grad << endl;
        float max_grad = 1.0;
        loss_grad = loss_grad.clip(-max_grad, max_grad);
        
        cout << "about to go backwords on layer 1" << endl;
        Matrix grad_output_1 = layer_1.backward(input, loss_grad);
        cout << "grad_output_1: \n" << grad_output_1 << endl;

        layer_1.clip_gradients(-max_grad, max_grad);
        layer_1.update_parameters(learning_rate);
        
        if (i % 1 == 0) {
            cout << "----------------" << endl;
            cout << "Iteration " << i << endl;
            cout << "output: " << output_1(0,0) << endl;
            cout << "Loss: " << loss << endl;
            cout << "target: " << target(0,0) << endl;
            cout << "----------------" << endl;
        }
    }
}
void test_learning_something_simple() {
    // Initialize layers with much smaller initial weights
    LinearLayer layer_1("test_linear_layer_1", 1, 2, 0.01);
    LinearLayer layer_2("test_linear_layer_2", 2, 1, 0.01);

    Matrix input(1, 1);
    input.set_row(0, {0.5});
    input.copy_to_gpu();

    Matrix target(1, 1);
    target(0, 0) = 0.5;
    target.copy_to_gpu();

    // Try a range of learning rates
    float learning_rate = 0.01;

    for (size_t i = 0; i < 5; ++i) {
        // cout << " right before forward" << endl;
        // cout << "input shape: " << input.shape().x << " " << input.shape().y << endl;
        Matrix output_1 = layer_1.forward(input);
        output_1.copy_to_cpu();
        // cout << " right after forward" << endl;
        Matrix output_2 = layer_2.forward(output_1);
        output_2.copy_to_cpu();

        target.copy_to_gpu();
        Matrix diff = output_2 - target;

        float loss = (diff * diff).sum_rows()(0, 0) / 2.0f;
        cout << "loss: " << loss << endl;
        cout << "output_2: \n" << output_2 << endl;

        
        Matrix loss_grad = diff;
        float max_grad = 1.0;
        loss_grad = loss_grad.clip(-max_grad, max_grad);
        
        cout << "about to go backwords on layer 2" << endl;
        Matrix grad_output_2 = layer_2.backward(output_1, loss_grad);
        cout << "grad_output_2: \n" << grad_output_2 << endl;

        cout << "about to go backwords on layer 1" << endl;
        Matrix grad_output_1 = layer_1.backward(input, grad_output_2);
        cout << "grad_output_1: \n" << grad_output_1 << endl;

        layer_1.clip_gradients(-max_grad, max_grad);
        layer_2.clip_gradients(-max_grad, max_grad);

        layer_1.update_parameters(learning_rate);
        layer_2.update_parameters(learning_rate);
        
        if (i % 1 == 0) {
            cout << "----------------" << endl;
            cout << "Iteration " << i << endl;
            cout << "output: " << output_2(0,0) << endl;
            cout << "loss: " << loss << endl;
            cout << "target: " << target(0,0) << endl;
            cout << "----------------" << endl;
        }
    }
}

int main(void)
{
    // test_sum_rows();
    // test_sum_cols();
    // test_matrix();
    // test_binary_cross_entropy_loss();
    // test_linear_layer();
    // test_one_layer();
    test_learning_something_simple();
    // test_transpose();
    // test_matrix_multiplication();
    return 0;
}