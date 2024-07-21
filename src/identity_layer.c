#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "model.h"

size_t get_identity_elm(size_t dim, size_t *width) {
  int elm = width[0];
  for (int d = 1; d < dim; d++) elm *= width[d];

  return elm;
}

void identity_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  size_t elm = layer->width[0] * batch;
  for (int d = 1; d < layer->dim; d++) elm *= layer->width[d];

  float *isig = input->signal;
  float *osig = layer->signal->signal;

  memcpy(osig, isig, elm);
}

void identity_backward(layer_t *layer, size_t batch) {
  size_t elm = layer->width[0] * batch;
  for (int d = 1; d < layer->dim; d++) elm *= layer->width[d];

  memcpy(layer->delta, layer->next->delta, elm);
}

void identity_print(layer_t *layer) {
  printf("identity_layer:\n");
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void identity_dump(layer_t *layer) {
}

void identity_layer(layer_t *layer, int step, size_t batch, int cmd) {
  switch (cmd) {
    case FORWARD  : identity_forward(layer, batch); break;
    case BACKWARD : identity_backward(layer, batch); break;
    case UPDATE   : break;
    case PRINT : identity_print(layer); break;
    case DUMP : identity_dump(layer); break;
    default : 
      fprintf(stderr, "Error: invalid command (%d) for identity_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_identity_layer(layer_t *model, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append relu layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  int ielm = tail->signal->width[0];
  for (int d = 1; d < tail->signal->dim; d++) ielm *= tail->signal->width[d];

  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = tail->signal->dim;
  sig->width      = tail->signal->width;
  sig->signal     = (float *)xmalloc(sizeof(float)*ielm*batch);

  float *delta    = (float *)xmalloc(sizeof(float)*ielm*batch);

  layer->dim      = tail->signal->dim;
  layer->width    = tail->signal->width;
  layer->fn_ptr   = identity_layer;
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
