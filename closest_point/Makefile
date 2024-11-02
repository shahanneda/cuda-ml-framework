all: cpu
# nvcc -ccbin gcc-9 -std=c++14 add.cu -o add -lstdc++

cpu: cpu.cpp
	gcc -o cpu cpu.cpp -lstdc++

clean:
	rm -f cpu