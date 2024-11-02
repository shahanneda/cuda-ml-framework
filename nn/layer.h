#include "matrix.h"

class Layer {

public:
    Layer(std::string name) : name_(name) {}
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& input, const Matrix& grad_output) = 0;
    std::string name() const { return name_; }

protected:
    std::string name_;
};