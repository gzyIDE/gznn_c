#include <string.h>
#include "util.h"
#include "model.h"
#include "operation.h"

size_t get_affine_elm(size_t dim, size_t *width) {
  int elm = width[0];
  for (int d = 1; d < dim; d++) elm *= width[d];

  return elm;
}

void affine_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  int och         = layer->width[0];
  float *isig     = input->signal;
  float *osig     = layer->signal->signal;

  // element width
  int elm = layer->width[1];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  // layer->weight: (elm, och)
  //matmul(isig, layer->weight, osig, batch, elm, elm, och, 0, 0);
  //vadd_batch(osig, layer->bias, osig, och, batch);

  gemm(isig, layer->weight, layer->bias, osig, 1.0, 1.0,
       batch, elm, elm, och, 0, 0, 1, 1);
}

void affine_backward(layer_t *layer, size_t batch) {
  int och = layer->width[0];

  // element width
  int elm = layer->width[1];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  // delta calculation
  float *delta_o = layer->next->delta; // delta from output node
  float *delta_i = layer->delta;       // delta to input node

  matmul(delta_o, layer->weight, delta_i, batch, och, elm, och, 0, 1);
}

void affine_update(layer_t *layer, int step, size_t batch) {
  signal_t *input = layer->prev->signal;
  size_t och      = layer->width[0];
  float *isig     = input->signal;

  // element width
  size_t elm = layer->width[1];
  for (int d = 2; d < layer->dim; d++) elm *= layer->width[d];

  // weight update
  float *delta_o = layer->next->delta; // delta from output node

  float *grad_w  = xcalloc(och*elm, sizeof(float));
  float *grad_b  = xcalloc(och, sizeof(float));
  float *isig_t  = xmalloc(sizeof(float)*elm*batch);

  // gradient calculation
  matmul(isig, delta_o, grad_w, batch, elm, batch, och, 1, 0);
  for (int b = 0; b < batch; b++) {
    vadd(grad_b, &delta_o[b*och], grad_b, och);
  }

  // Update with Adam
  float b1   = layer->beta1;
  float b2   = layer->beta2;
  float rate = layer->learning_rate;
  adam(layer->wm, layer->wv, grad_w, layer->weight, b1, b2, rate, step, elm*och);
  adam(layer->bm, layer->bv, grad_b, layer->bias, b1, b2, rate, step, och);

  free(grad_w);
  free(grad_b);
  free(isig_t);
}

void affine_print(layer_t *layer) {
  printf("affine_layer:\n");
  printf("  weight[%ld]:\n", layer->dim);
  printf("    ( ");
  for (int i = 0; i < layer->dim; i++) {
    printf("%ld ", layer->width[i]);
  }
  printf(")\n");
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void affine_dump(layer_t *layer) {
}

void affine_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("affine_layer\n");

  switch (cmd) {
    case FORWARD  : affine_forward(layer, batch); break;
    case BACKWARD : affine_backward(layer, batch); break;
    case UPDATE   : affine_update(layer, step, batch); break;
    case PRINT    : affine_print(layer); break;
    case DUMP     : affine_dump(layer); break;
    default : 
      fprintf(stderr, "Error: invalid command (%d) for affine_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_affine_layer(layer_t *model, param_t p, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append affine layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  size_t ielm     = get_affine_elm(tail->signal->dim, tail->signal->width);

  size_t *width   = (size_t *)xmalloc(sizeof(size_t)*(tail->signal->dim+1));
  width[0]        = p.och;
  for (int i = 0; i < tail->signal->dim; i++) {
    width[i+1] = tail->signal->width[i];
  }

  size_t *osigw   = (size_t *)xmalloc(sizeof(size_t));
  osigw[0]        = p.och;

  int wsize       = p.och * ielm;
  int wdim        = tail->signal->dim + 1;
  float *weight   = (float *)xmalloc(sizeof(float)*wsize);
  //float wscale    = 1.0 / sqrt(ielm);
  float wscale    = sqrt( 2.0 / ielm);
  init_weight(wdim, width, weight, wscale);


  int bsize       = p.och;
  float *bias     = (float *)xmalloc(sizeof(float)*bsize);
  init_bias(bsize, bias);

  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = 1;
  sig->width      = osigw;
  sig->signal     = (float *)xmalloc(sizeof(float)*p.och*batch);

  float *delta    = (float *)xmalloc(sizeof(float)*ielm*batch);

  layer->dim      = wdim;
  layer->width    = width;
  layer->fn_ptr   = affine_layer;
  layer->stride   = 0;
  layer->pad      = 0;
  layer->weight   = weight;
  layer->signal   = sig;
  layer->bias     = bias;
  layer->delta    = delta;
  layer->wv       = (float *)xcalloc(wsize, sizeof(float));
  layer->wm       = (float *)xcalloc(wsize, sizeof(float));
  layer->bv       = (float *)xcalloc(bsize, sizeof(float));
  layer->bm       = (float *)xcalloc(bsize, sizeof(float));
  layer->learning_rate = p.learning_rate;
  layer->beta1    = p.adam_beta1;
  layer->beta2    = p.adam_beta2;
  layer->strsize  = 0;
  layer->string   = NULL;

  tail->next = layer;
  layer->prev = tail;
  layer->next = NULL;

  return layer;
}
