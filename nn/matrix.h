#pragma once
#include "shape.h"
#include <memory>

class Matrix {
    public:
        Matrix(Shape shape);
        Matrix(size_t rows, size_t cols);
        ~Matrix();
        Shape shape() const;

        std::shared_ptr<float> cpu_data_ptr;
        std::shared_ptr<float> gpu_data_ptr;
        void allocate_memory();

        int total_size() const;
        float& operator()(size_t row, size_t col);
        const float& operator()(size_t row, size_t col) const;

        float& operator[](size_t index);
        const float& operator[](size_t index) const;

        void copy_to_gpu();
        void copy_to_cpu();

    private:
        Shape shape_;
        float* data_;

        void allocate_cpu_memory();
        void allocate_gpu_memory();

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};