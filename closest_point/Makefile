all: cpu gpu

cpu: cpu.cpp
	gcc -o cpu cpu.cpp -lstdc++
gpu: gpu.cu
	nvcc -g -G -ccbin gcc-9 -std=c++14 gpu.cu -o gpu -lstdc++

clean:
	rm -f cpu