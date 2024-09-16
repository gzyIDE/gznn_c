vpath %.c ./src
#CFLAGS = -lm -O3 -mavx2 -march=native
CFLAGS = -lm -O3 -march=native
CFLAGS += -fopenmp -lgomp
INCDIR = -I./include -I./config

SRCS = main.c           \
			 load_cifar10.c   \
			 load_mnist.c     \
			 bmp.c            \
			 model.c          \
			 operation.c      \
			 affine_layer.c   \
			 softmax_layer.c  \
			 util.c           \
			 cross_entropy.c  \
			 relu_layer.c     \
			 identity_layer.c \
			 conv_layer.c     \
			 input_layer.c    \
			 maxpool_layer.c


main : $(SRCS)
	gcc $(INCDIR) $^ $(CFLAGS) -o $@

clean:
	rm -f main
