#ifndef _model_h_included_
#define _model_h_included_

#include <stdint.h>
#include <stddef.h>

//***** Command
#define FORWARD   0
#define BACKWARD  1
#define UPDATE    2
#define PRINT     3
#define DUMP      4

//***** Dataset definition (label and data pair)
typedef struct {
  uint8_t   label;  // Data Label
  size_t    dim;    // data dimension
  size_t    *width; // pointer to the width vector
  uint8_t   *data;  // Pointer to data
} dataset_t;


//***** Signal transfered between layers
typedef struct {
  size_t    dim;     // data dimension
  size_t    *width;  // width
  float     *signal; // signal between layers
} signal_t;


//***** Parameter
typedef struct {
  size_t och;
  size_t fsize;   // for convolution
  size_t pad;     // for convolution
  size_t stride;  // for convolution
  float  adam_beta1;
  float  adam_beta2;
  float  learning_rate;
} param_t;


//***** Layer definition
typedef struct layer_t {
  struct layer_t  *next;   // pointer to next layer
  struct layer_t  *prev;   // pointer to previous layer
  void (*fn_ptr)(struct layer_t *layer, int step, size_t batch, int cmd); // function pointer
  size_t          dim;     // kernel/weight dimension
  size_t          *width;  // pointer to the width vector
  size_t          stride;  // stride for convolution
  size_t          pad;     // padding for convolution
  float           *weight; // pointer to weight/kernels
  float           *bias;   // pointer to bias
  signal_t        *signal; // signal to next layer
  float           *delta;  // delta for backpropagation
  float           learning_rate;
  float           beta1;    // coefficient1 of Adam
  float           beta2;    // coefficient2 of Adam
  float           *wv;      // first moment of Adam (weight)
  float           *wm;      // second moment of Adam (weight)
  float           *bv;      // first moment of Adam (bias)
  float           *bm;      // second moment of Adam (bias)
  size_t          strsize;  // output string size
  char            *string;  // output string
} layer_t;


//***** function definition
void forward(layer_t *model, dataset_t *data, size_t batch);
float backward(layer_t *model, dataset_t *data, size_t batch);
void update(layer_t *model, dataset_t *data, int step, size_t batch);
float *normalize(dataset_t *data, float scale, size_t w, size_t batch);
float *label2vec(dataset_t *data, size_t w, size_t batch);
layer_t *find_first(layer_t *model);
layer_t *find_last(layer_t *model);
void init_weight(size_t dim, size_t *width, float *weight, float scale);
void init_bias(size_t width, float *bias);
void print_layer(layer_t *model);
void dump_model(layer_t *model);
#endif //_model_h_included_
