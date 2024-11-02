#pragma once
#include "shape.h"
#include <memory>

class Matrix {
    public:
        Matrix(Shape shape);
        Matrix(size_t rows, size_t cols);
        ~Matrix();
        Shape shape() const;

        std::shared_ptr<float*> cpu_data_ptr;
        std::shared_ptr<float*> gpu_data_ptr;
        void allocate_memory();
    private:
        Shape shape_;
        float* data_;

        void allocate_cpu_memory();
        void allocate_gpu_memory();
};