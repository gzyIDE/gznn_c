#include <string.h>
#include "model.h"
#include "operation.h"

void fc_forward(layer_t *layer, float *input) {
  // output channel
  int och = layer->width[0];

  // element width
  int elm = layer->width[1];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  for (int ch = 0; ch < och; ch++) {
    float *weight = &layer->weight[ch*elm];
    layer->signal[ch] = dot(weight, input, elm) + layer->bias[ch];
  }
}

void fc_backward(layer_t *layer) {
  // output channel
  int och = layer->width[0];

  // element width
  int elm = layer->width[1];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  // delta calculation
  float *delta_o = layer->next->delta; // delta from output node
  float *delta_i = layer->delta;       // delta to input node
  memset(delta_i, 0, sizeof(float)*elm);
  for (int ch = 0; ch < och; ch++) {
    float *weight = &layer->weight[ch*elm];
    saxpy(delta_o[ch], weight, delta_i, delta_i, elm);
  }
}

void fc_update(layer_t *layer, float *input) {
}

void fc_layer(layer_t *layer, float *input, int dir) {
  //printf("fc_layer\n");

  if ( dir ) {
    // backward
    fc_backward(layer);
    fc_update(layer, input);
  } else {
    // forward
    fc_forward(layer, input);
  }
}
