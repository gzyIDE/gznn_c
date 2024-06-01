vpath %.c ./src
main : main.c cifar10.c bmp.c model.c operation.c fc_layer.c softmax_layer.c util.c cross_entropy.c
	gcc -I./include $^ -lm -o $@

clean:
	rm -f main
