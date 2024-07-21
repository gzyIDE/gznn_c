#define EPOCH      1
#define BATCH_SIZE 10
#define TRAIN_SIZE 60000
#define TEST_SIZE  10000
param_t param;

//***** Data load
char train_path[] = "./dataset/mnist/mnist_train.csv";
char test_path[] = "./dataset/mnist/mnist_test.csv";
train_data = (dataset_t *)xmalloc(sizeof(dataset_t)*TRAIN_SIZE);
test_data  = (dataset_t *)xmalloc(sizeof(dataset_t)*TEST_SIZE);
load_mnist(train_path, train_data, TRAIN_SIZE);
load_mnist(test_path, test_data, TEST_SIZE);

//***** Layer definition
// input layer
model = generate_input_layer(1, 28, 28, BATCH_SIZE);

param.och = 10;
param.adam_beta1    = 0.9;
param.adam_beta2    = 0.999;
param.learning_rate = 0.001;
add_affine_layer(model, param, BATCH_SIZE);

output_layer = add_softmax_layer(model, BATCH_SIZE);

odim = 10;
