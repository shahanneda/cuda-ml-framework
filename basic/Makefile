all: hello_world add

hello_world: hello_world.cu
	# nvcc -std=c++11 hello_world.cu -o hello_world
	nvcc -ccbin gcc-9 -std=c++14 hello_world.cu -o hello_world -lstdc++

add: add.cu
	nvcc -ccbin gcc-9 -std=c++14 add.cu -o add -lstdc++

clean:
	rm -f hello_world