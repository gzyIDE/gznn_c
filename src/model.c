#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "operation.h"
#include "cross_entropy.h"

void forward(layer_t *model, dataset_t *data) {
  float *input_data;
  int   elm = data->width[0];
  for (int i = 1; i < data->dim; i++) elm *= data->width[i];

  // input data conversion
  input_data = (float *)xmalloc(sizeof(float)*elm);
  for (int i = 0; i < elm; i++) {
    input_data[i] = (float)data->data[i];
  }

  layer_t *current = model;
  do {
    float *input;
    if (current->prev) {
      input = current->prev->signal;
    } else {
      input = input_data;
    }

    if ( current->fn_ptr == NULL ) {
      // output layer
      memcpy(current->signal, current->prev->signal, sizeof(float)*current->width[0]);
    } else {
      current->fn_ptr(current, input, FORWARD);
    }

  } while (current = current->next);

  free(input_data);
}

float backward(layer_t *model, dataset_t *data) {
  // find output layer
  layer_t *current = model;
  while (current->next) current = current->next;

  // output delta calculation
  int tvec_w = current->width[0];
  float *tvec = (float *)xmalloc(sizeof(float) * tvec_w);
  for ( int i = 0; i < tvec_w; i++ ) {
    tvec[i] = (i == data->label) ? 1.0 : 0.0;
  }
  float loss = cross_entropy(tvec, current->signal, tvec_w);
  cross_entropy_backward(tvec, current->signal, current->delta, tvec_w);


  // back propagation
  current = current->prev;
  do {
    current->fn_ptr(current, current->prev->signal, BACKWARD);
    current = current->prev;
  } while (current = current->prev);

  return loss;
}

void train(layer_t *model, dataset_t *data) {
}
