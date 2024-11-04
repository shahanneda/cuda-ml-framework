A simple machine learning framework built completly from scatch in C++ and CUDA with no other dependencies for educational purposes.

Currently supports:
- Matrix Operations: Addition, Subtraction, Multiplication (scaler and matrix), Transposition, Summing along rows and along columns, clipping values
- Fully Connected Layers of any input and output size
- Activation Layers: ReLU, Sigmoid
- Loss Layers: Binary Cross Entropy Loss
(all layers have both forward and backward pass implemented)

TODO:
- Convolution, pooling, softmax, drop out, batch normalization layers
- Learning rate scheduling
- Basic plotting of the loss curve


Here is an example of how you can use it:

```c++
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
```


### Compiling:
```bash
cd nn && make && ./nn
```
See main.cu for example usages.
