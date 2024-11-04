# CUDA compiler
NVCC = nvcc
# C++ compiler to use with CUDA
CCBIN = gcc-9
# Compiler and linker flags
NVCCFLAGS = -ccbin $(CCBIN) -std=c++14 -g -G
LDFLAGS = -lstdc++ -lm

# Target executable
TARGET = nn

# Source files
SOURCES = main.cu shape.cu matrix.cu binary_cross_entropy_loss.cu linear_layer.cu sigmoid_activation_layer.cu relu_activation_layer.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile CUDA source files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean