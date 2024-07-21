#define EPOCH      1
#define BATCH_SIZE 10
#define TRAIN_SIZE 50000
#define TEST_SIZE  10000
param_t param;

//***** Data load
char train_path[][128] = {
  "./dataset/cifar-10-batches-bin/data_batch_1.bin",
  "./dataset/cifar-10-batches-bin/data_batch_2.bin",
  "./dataset/cifar-10-batches-bin/data_batch_3.bin",
  "./dataset/cifar-10-batches-bin/data_batch_4.bin",
  "./dataset/cifar-10-batches-bin/data_batch_5.bin"};
char test_path[] = "./dataset/cifar-10-batches-bin/test_batch.bin";
train_data = (dataset_t *)xmalloc(sizeof(dataset_t)*TRAIN_SIZE);
test_data  = (dataset_t *)xmalloc(sizeof(dataset_t)*TEST_SIZE);
for (int i = 0; i < 5; i++) {
  load_cifar10(train_path[i], &train_data[10000*i], 10000);
}
load_cifar10(test_path, test_data, TEST_SIZE);

//***** Layer definition
// input layer
model = generate_input_layer(3, 32, 32, BATCH_SIZE);

// layer1: Convolution Layer
param.och           = 20;
param.fsize         = 3;
param.pad           = 1;
param.stride        = 1;
param.adam_beta1    = 0.9;
param.adam_beta2    = 0.999;
param.learning_rate = 0.001;
add_conv_layer(model, param, BATCH_SIZE);

// layer2: maxpool layer
param.fsize         = 2;
param.pad           = 0;
param.stride        = 2;
add_maxpool_layer(model, param, BATCH_SIZE);

// layer3: Convolution Layer
param.och           = 10;
param.fsize         = 3;
param.pad           = 1;
param.stride        = 1;
param.adam_beta1    = 0.9;
param.adam_beta2    = 0.999;
param.learning_rate = 0.001;
add_conv_layer(model, param, BATCH_SIZE);

// layer3 :Relu
add_relu_layer(model, BATCH_SIZE);

// layer4: Fully connect Layer
param.och           = 10;
param.adam_beta1    = 0.9;
param.adam_beta2    = 0.999;
param.learning_rate = 0.001;
add_affine_layer(model, param, BATCH_SIZE);

// layer4: Softmax Layer
output_layer = add_softmax_layer(model, BATCH_SIZE);

odim = 10;
