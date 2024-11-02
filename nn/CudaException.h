#include <exception>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>

class CudaException : public std::exception {
public:
    CudaException(const std::string& message) : msg_(message) {}
    virtual const char* what() const noexcept override {
        return msg_.c_str();
    }
    static void throw_if_error(const std::string& message) {
        cudaError_t error  = cudaGetLastError();
        if (error != cudaSuccess){
            std::cerr <<  message  << ":" << error;
            throw CudaException(message);
        }
    }
private:
    std::string msg_;

};
