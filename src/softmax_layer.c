#include "model.h"
#include "operation.h"

void softmax_forward(layer_t *layer, float *input) {
  softmax(input, layer->signal, layer->width[0]);
}

void softmax_backward(layer_t *layer, float *input) {
  int elm = layer->width[0];

  // delta calculation
  float *delta_o = layer->next->delta;
  for (int i = 0; i < elm; i++) {
    float sig = layer->signal[i];
    layer->delta[i] = layer->delta[i] * (1.0 - sig) * sig;
  }
}

void softmax_layer(layer_t *layer, float *input, int dir) {
  //printf("softmax_layer\n");

  if ( dir ) {
    softmax_backward(layer, input);
  } else {
    softmax_forward(layer, input);
  }
}
