all: add

add: add.cu
	nvcc -ccbin gcc-9 -std=c++14 add.cu -o add -lstdc++

clean:
	rm -f add