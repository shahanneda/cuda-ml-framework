#pragma once
#include <cuda.h>


struct Shape{
    size_t x, y;
    Shape(size_t x = 1, size_t y = 1);
};