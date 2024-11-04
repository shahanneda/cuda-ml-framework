#include <iostream>
#include <stdio.h>

#include "shape.h"
#include "matrix.h"
#include "binary_cross_entropy_loss.h"
#include "linear_layer.h"
#include "sigmoid_activation_layer.h"
#include "relu_activation_layer.h"
using namespace std;

void test_binary_cross_entropy_loss() {
    Matrix y_true = Matrix(10, 1);
    Matrix y_pred = Matrix(10, 1);
    y_true.set_col(0, {0, 0, 1, 0, 1, 0, 1, 1, 1, 0});
    y_pred.set_col(0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
    cout << "y_pred: \n" << y_pred << endl;
    cout << "y_true: \n" << y_true << endl;
    y_true.copy_to_gpu();
    y_pred.copy_to_gpu();
    BinaryCrossEntropyLoss loss("binary_cross_entropy_loss");
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

void test_copying_to_gpu_and_back() {
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


void test_sigmoid_activation_layer(){
    SigmoidActivationLayer layer("sigmoid_activation_layer");
    Matrix input(1, 3);
    input.set_row(0, {0.5, 1.0, 2.0});
    input.copy_to_gpu();

    cout << "input to sigmoid: \n" << input << endl;
    Matrix output = layer.forward(input);
    output.copy_to_cpu();
    cout << "output: \n" << output << endl;

    Matrix grad_output(1, 3);
    grad_output.set_row(0, {1, 1, 1});
    grad_output.copy_to_gpu();

    Matrix grad = layer.backward(input, grad_output);
    grad.copy_to_cpu();
    cout << "grad: \n" << grad << endl;
    // Expected output
    // output:  tensor([0.6225, 0.7311, 0.8808], grad_fn=<SigmoidBackward0>)
    // grad:  tensor([0.2350, 0.1966, 0.1050])
}

void test_relu_activation_layer() {
    ReluActivationLayer layer("test_relu_activation_layer");
    Matrix input(1, 3);

    input.set_row(0, {-0.5, -8.0, 2.0});
    cout << "input: \n" << input << endl;
    input.copy_to_gpu();

    Matrix output = layer.forward(input);
    output.copy_to_cpu();
    cout << "output: \n" << output << endl;

    Matrix grad_output(1, 3);
    grad_output.set_row(0, {1, 1, 1});
    grad_output.copy_to_gpu();

    Matrix grad = layer.backward(input, grad_output);
    grad.copy_to_cpu();
    cout << "grad: \n" << grad << endl;
}


void test_memorizing_value() {
    // This tests a simple network that learns to memorize a value
    // Fully Connected Layer -> ReLU -> Fully Connected Layer -> Sigmoid

    // Initialize layers with much smaller initial weights
    LinearLayer layer_1("linear_layer_1", 1, 5, 0.01);
    ReluActivationLayer layer_1_activation("relu_activation_layer");

    LinearLayer layer_2("linear_layer_2", 5, 1, 0.01);
    SigmoidActivationLayer layer_2_activation("sigmoid_activation_layer");

    Matrix input(1, 1);
    input.set_row(0, {0.5});
    input.copy_to_gpu();

    Matrix target(1, 1);
    target(0, 0) = 0.78;
    target.copy_to_gpu();

    // Try a range of learning rates
    float learning_rate = 0.01;

    for (size_t i = 0; i < 10; ++i) {
        // forward pass
        Matrix output_1 = layer_1.forward(input);
        Matrix output_1_activation = layer_1_activation.forward(output_1);
        Matrix output_2 = layer_2.forward(output_1_activation);
        Matrix output_2_activation = layer_2_activation.forward(output_2);
        target.copy_to_gpu();
        Matrix diff = output_2_activation - target;

        // calculate loss
        float loss = (diff * diff).sum_rows()(0, 0) / 2.0f;
        Matrix loss_grad = diff;
        float max_grad = 1.0;
        loss_grad = loss_grad.clip(-max_grad, max_grad);

        // backward pass
        Matrix grad_output_3 = layer_2_activation.backward(output_2, loss_grad);
        Matrix grad_output_2 = layer_2.backward(output_1_activation, grad_output_3);
        Matrix grad_output_1 = layer_1_activation.backward(output_1, grad_output_2);
        Matrix grad_output_1_raw = layer_1.backward(input, grad_output_1);

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

void test_simple_classifier(){
    // This trains a simple classifer which outputs 1 if the input is in the top right or bottom left quadrants, but 0 otherwise
    // (batch_size, 2) -> Fully Connected Layer (2, 5) -> Relu -> Fully Connected Layer (5, 1) -> Sigmoid

    // Generate data set
    Matrix inputs (50, 2);
    Matrix targets (50, 1);
    for (size_t i = 0; i < inputs.shape().x; ++i) {
        // Generates a random number between -1 and 1
        float x = (((float)rand() / RAND_MAX) - 0.5)*2;
        float y = (((float)rand() / RAND_MAX) - 0.5)*2;
        inputs(i, 0) = x;
        inputs(i, 1) = y;
        targets(i, 0) = (x > 0.5 && y > 0.5) || (x < 0.5 && y < 0.5) ? 1 : 0;
    }
    inputs.copy_to_gpu();
    targets.copy_to_gpu();

    // Initialize layers
    LinearLayer layer_1("linear_layer_1", 2, 5, 0.01);
    ReluActivationLayer layer_1_activation("relu_activation_layer");

    LinearLayer layer_2("linear_layer_2", 5, 1, 0.01);
    SigmoidActivationLayer layer_2_activation("sigmoid_activation_layer");

    BinaryCrossEntropyLoss loss("binary_cross_entropy_loss");

    float learning_rate = 0.01;
    float max_grad = 1.0;

    for (size_t epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        Matrix output_1 = layer_1.forward(inputs);
        Matrix output_1_activation = layer_1_activation.forward(output_1);
        Matrix output_2 = layer_2.forward(output_1_activation);
        Matrix output_2_activation = layer_2_activation.forward(output_2);

        // Calculate loss
        float loss_value = loss.forward(targets, output_2_activation);

        // Backward pass
        Matrix loss_grad = loss.backward(targets, output_2_activation);
        Matrix grad_output_2 = layer_2_activation.backward(output_2, loss_grad);
        Matrix grad_output_1 = layer_2.backward(output_1_activation, grad_output_2);
        Matrix grad_output_1_raw = layer_1_activation.backward(output_1, grad_output_1);
        layer_1.clip_gradients(-max_grad, max_grad);
        layer_2.clip_gradients(-max_grad, max_grad);

        layer_1.update_parameters(learning_rate);
        layer_2.update_parameters(learning_rate);

        if (epoch % 10 == 0) {
            cout << "----------------" << endl;
            cout << "Epoch " << epoch << endl;
            cout << "loss: " << loss_value << endl;
            cout << "----------------" << endl;
        }
    }
}

int main(void)
{
    // test_sum_rows();
    // test_sum_cols();
    // test_copying_to_gpu_and_back();
    // test_matrix();
    // test_binary_cross_entropy_loss();
    // test_linear_layer();
    // test_one_layer();
    // test_memorizing_value();
    // test_transpose();
    // test_matrix_multiplication();
    // test_sigmoid_activation_layer();
    // test_relu_activation_layer();
    test_simple_classifier();
    return 0;
}