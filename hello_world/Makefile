all: hello_world

hello_world: hello_world.cu
	nvcc -ccbin gcc-9 -std=c++14 hello_world.cu -o hello_world -lstdc++
clean:
	rm -f hello_world