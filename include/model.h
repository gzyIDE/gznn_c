#ifndef _model_h_included_
#define _model_h_included_

#include <stdint.h>

#define FORWARD   0
#define BACKWARD  1

//***** Dataset definition (label and data pair)
typedef struct {
  uint8_t   label;  // Data Label
  uint8_t   dim;    // data dimension
  int       *width; // pointer to the width vector
  uint8_t   *data;  // Pointer to data
} dataset_t;


//***** Layer definition
typedef struct layer_t {
  struct layer_t  *next;   // pointer to next layer
  struct layer_t  *prev;   // pointer to previous layer
  void (*fn_ptr)(struct layer_t *layer, float *input, int dir); // function pointer
  uint8_t         dim;     // kernel/weight dimension
  int             *width;  // pointer to the width vector
  int             *stride; // pointer to the stride vector for convolution
  float           *weight; // pointer to weight/kernels
  float           *bias;   // pointer to bias
  float           *signal; // signal to next layer
  float           *delta;  // delta for backpropagation
} layer_t;


//***** Signal transfered between layers
typedef struct {
  uint8_t   dim;    // data dimension
  int       *width; // width
  float     *sig;   // signal between layers
} signal_t;


//***** function definition
void forward(layer_t *model, dataset_t *data);
float backward(layer_t *model, dataset_t *data);
#endif //_model_h_included_
