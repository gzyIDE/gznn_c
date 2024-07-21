#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "operation.h"

void softmax_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  size_t elm      = layer->width[0];
  for (int d = 1; d < layer->dim; d++) elm *= layer->width[d];

  float *isig = input->signal;
  float *osig = layer->signal->signal;

  for (int b = 0; b < batch; b++) {
    softmax(&isig[b*elm], &osig[b*elm], elm);
  }
}

void softmax_backward(layer_t *layer, size_t batch) {
  int elm     = layer->width[0];
  for (int d = 1; d < layer->dim; d++) elm *= layer->width[d];

  // delta calculation
  float *delta_o = layer->next->delta;
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < elm; i++) {
      float osig = layer->signal->signal[b*elm+i];
      layer->delta[b*elm+i] = layer->delta[b*elm+i] * (1.0 - osig) * osig;
    }
  }
}

void softmax_with_loss_delta(layer_t *layer, float *t, size_t batch) {
  int elm     = layer->width[0];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  float *osig = layer->signal->signal;

  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < elm; i++) {
      layer->delta[b*elm+i] = (osig[b*elm+i] - t[b*elm+i])/(float)batch;
    }
  }
}

void softmax_print(layer_t *layer) {
  printf("softmax_layer:\n");
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void softmax_dump(layer_t *layer) {
}

void softmax_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("softmax_layer\n");

  switch (cmd) {
    case FORWARD  : softmax_forward(layer, batch); break;
    case BACKWARD : softmax_backward(layer, batch); break;
    case UPDATE   : break;
    case PRINT    : softmax_print(layer); break;
    case DUMP     : softmax_dump(layer); break;
    default : 
      fprintf(stderr, "Error: invalid command (%d) for softmax_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_softmax_layer(layer_t *model, param_t p, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append softmax layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  if ( tail->signal->dim != 1 ) {
    fprintf(stderr, "Error: failed to append softmax layer\n");
    fprintf(stderr, "       Invalid input dimension (%ld)\n", tail->signal->dim);
    exit(EXIT_FAILURE);
  }

  int ielm        = tail->signal->width[0];

  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = 1;
  sig->width      = tail->signal->width;
  sig->signal     = (float *)xmalloc(sizeof(float)*ielm*batch);

  float *delta    = (float *)xmalloc(sizeof(float)*ielm*batch);

  layer->dim      = 1;
  layer->width    = tail->signal->width;
  layer->fn_ptr   = softmax_layer;
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
