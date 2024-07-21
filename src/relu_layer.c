#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "operation.h"

size_t get_relu_elm(size_t dim, size_t *width) {
  int elm = width[0];
  for (int d = 1; d < dim; d++) elm *= width[d];

  return elm;
}

void relu_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  size_t elm      = get_relu_elm(layer->dim, layer->width) * batch;

  float *isig     = input->signal;
  float *osig     = layer->signal->signal;

  relu(isig, isig, osig, elm);
}

void relu_backward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  size_t elm      = get_relu_elm(layer->dim, layer->width) * batch;

  float *delta_o  = layer->next->delta;
  float *isig     = input->signal;

  relu(isig, delta_o, layer->delta, elm);
}

void relu_print(layer_t *layer) {
  printf("relu_layer:\n");
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void relu_dump(layer_t *layer) {
}

void relu_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("relu_layer\n");

  switch (cmd) {
    case FORWARD  : relu_forward(layer, batch); break;
    case BACKWARD : relu_backward(layer, batch); break;
    case UPDATE   : break;
    case PRINT    : relu_print(layer); break;
    case DUMP     : relu_dump(layer); break;
    default       : 
      fprintf(stderr, "Error: invalid command (%d) for relu_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_relu_layer(layer_t *model, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append relu layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  size_t *width   = (size_t *)xmalloc(sizeof(size_t));
  size_t ielm     = get_relu_elm(tail->signal->dim, tail->signal->width);

  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = tail->signal->dim;
  sig->width      = tail->signal->width;
  sig->signal     = (float *)xmalloc(sizeof(float)*ielm*batch);

  float *delta    = (float *)xmalloc(sizeof(float)*ielm*batch);

  layer->dim      = tail->signal->dim;
  layer->width    = tail->signal->width;
  layer->fn_ptr   = relu_layer;
  layer->stride   = 0;
  layer->pad      = 0;
  layer->weight   = NULL;
  layer->signal   = sig;
  layer->bias     = NULL;
  layer->delta    = delta;
  layer->wv       = NULL;
  layer->wm       = NULL;
  layer->bv       = NULL;
  layer->bm       = NULL;
  layer->learning_rate = 0.0;
  layer->beta1    = 0.0;
  layer->beta2    = 0.0;
  layer->strsize  = 0;
  layer->string   = NULL;

  tail->next = layer;
  layer->prev = tail;
  layer->next = NULL;

  return layer;
}
