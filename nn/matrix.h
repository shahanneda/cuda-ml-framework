#pragma once
#include "shape.h"
#include <memory>
#include <vector>

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

        void set_row(size_t row, const std::vector<float>& values);
        void set_col(size_t col, const std::vector<float>& values);

        void copy_to_gpu();
        void copy_to_cpu();
        uint32_t rows() const { return shape_.x; }
        uint32_t cols() const { return shape_.y; }

        Matrix T() const;
        Matrix operator*(const Matrix& other) const;

    private:
        Shape shape_;
        float* data_;

        void allocate_cpu_memory();
        void allocate_gpu_memory();

        friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};