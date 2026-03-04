NVCC = nvcc
FLAGS = -O3

all:
	$(NVCC) main.cu kernels.cu -o benchmark $(FLAGS)

clean:
	rm -f benchmark