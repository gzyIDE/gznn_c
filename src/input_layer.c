#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "operation.h"

void input_print(layer_t *layer) {
  printf("input_layer:\n");
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void input_dump(layer_t *layer) {
}

void input_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("input layer\n");
  switch (cmd) {
    case FORWARD: break;
    case BACKWARD: break;
    case UPDATE : break;
    case PRINT : input_print(layer); break;
    case DUMP : input_dump(layer); break;
    default :
      fprintf(stderr, "Error: invalid command (%d) for input_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *generate_input_layer(int ch, int row, int col, int batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  signal_t *signal = (signal_t *)xmalloc(sizeof(signal_t));
  size_t   *width  = (size_t *)xmalloc(sizeof(size_t)*3);
  width[0]         = ch;
  width[1]         = row;
  width[2]         = col;

  signal->dim      = 3;
  signal->width    = width;
  signal->signal   = NULL;

  layer->dim      = 0;
  layer->fn_ptr   = input_layer;
  layer->width    = NULL;
  layer->stride   = 0;
  layer->pad      = 0;
  layer->weight   = NULL;
  layer->signal   = signal;
  layer->wv       = NULL;
  layer->wm       = NULL;
  layer->bv       = NULL;
  layer->bm       = NULL;
  layer->learning_rate = 0.0;
  layer->beta1    = 0.0;
  layer->beta2    = 0.0;
  layer->strsize  = 0;
  layer->string   = NULL;

  layer->prev = NULL;
  layer->next = NULL;

  return layer;
}
