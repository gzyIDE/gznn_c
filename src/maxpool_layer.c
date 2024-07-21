#include <string.h>
#include <stdlib.h>
#include <float.h>
#include "util.h"
#include "model.h"
#include "operation.h"

size_t get_maxpool_osize(size_t isize, size_t fsize, size_t pad, size_t stride) {
  return (isize + 2 * pad - fsize) / stride + 1;
}

void maxpool_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  int ich         = layer->width[0];
  int fsize       = layer->width[1];
  int pad         = layer->pad;
  int stride      = layer->stride;
  int irow        = input->width[1];
  int icol        = input->width[2];
  int och         = layer->signal->width[0];
  int orow        = layer->signal->width[1];
  int ocol        = layer->signal->width[2];

  int i, j, k, l;
  #pragma omp parallel for private(i,j,k,l)
  for (int b = 0; b < batch; b++) {
    float *in  = &input->signal[b*irow*icol*ich];
    float *out = &layer->signal->signal[b*orow*ocol*och];

    for (int ch = 0; ch < och; ch++) {
      for (i = 0; i < orow; i++) {
        for (j = 0; j < ocol; j++) {
          float max  = FLT_MIN;
          int irowst = -pad + i * stride;
          int icolst = -pad + j * stride;
          for (k = 0; k < fsize; k++) {
            for (l = 0; l < fsize; l++) {
              int rowidx = irowst + k;
              int colidx = icolst + l;
              int iidx   = ch*irow*icol+rowidx*icol+colidx;
              if ((rowidx >= 0) && (rowidx < irow) && (colidx >= 0) && (colidx < icol)) {
                float d = in[iidx];
                if ( d > max ) {
                  max = d;
                }
              }
            }
          }
          out[ch*orow*ocol+i*ocol+j] = max;
        }
      }
    }
  }
}

void maxpool_backward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  int ich         = layer->width[0];
  int fsize       = layer->width[1];
  int pad         = layer->pad;
  int stride      = layer->stride;
  int irow        = input->width[1];
  int icol        = input->width[2];
  int och         = layer->signal->width[0];
  int orow        = layer->signal->width[1];
  int ocol        = layer->signal->width[2];

  memset(layer->delta, 0, sizeof(float)*irow*icol*ich*batch);

  int i, j, k, l;
  #pragma omp parallel for private(i,j,k,l)
  for (int b = 0; b < batch; b++) {
    float *in      = &input->signal[b*irow*icol*ich];
    float *out     = &layer->signal->signal[b*orow*ocol*och];
    float *delta_i = &layer->delta[b*irow*icol*ich];
    float *delta_o = &layer->next->delta[b*orow*ocol*och];

    for (int ch = 0; ch < och; ch++) {
      for (i = 0; i < orow; i++) {
        for (j = 0; j < ocol; j++) {
          int oidx   = ch*orow*ocol+i*ocol+j;
          int irowst = -pad + i * stride;
          int icolst = -pad + j * stride;
          for (k = 0; k < fsize; k++) {
            for (l = 0; l < fsize; l++) {
              int rowidx = irowst + k;
              int colidx = icolst + l;
              int iidx   = ch*irow*icol+rowidx*icol+colidx;
              if ((rowidx >= 0) && (rowidx < irow) && (colidx >= 0) && (colidx < icol)) {
                float d = in[iidx];
                if ( in[iidx] == out[oidx] ) {
                  delta_i[iidx] += delta_o[oidx];
                }
              }
            }
          }
        }
      }
    }
  }
}

void maxpool_print(layer_t *layer) {
  printf("maxpool_layer:\n");
  printf("  size  : %ld\n", layer->width[1]);
  printf("  stride: %ld\n", layer->stride);
  printf("  pad   : %ld\n", layer->pad);
  printf("  signal[%ld]:\n", layer->signal->dim);
  printf("    ( ");
  for (int i = 0; i < layer->signal->dim; i++) {
    printf("%ld ", layer->signal->width[i]);
  }
  printf(")\n");
}

void maxpool_dump(layer_t *layer) {
}

void maxpool_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("maxpool_layer\n");

  switch (cmd) {
    case FORWARD  : maxpool_forward(layer, batch); break;
    case BACKWARD : maxpool_backward(layer, batch); break;
    case UPDATE   : break;
    case PRINT    : maxpool_print(layer); break;
    case DUMP     : maxpool_dump(layer); break;
    default : 
      fprintf(stderr, "Error: invalid command (%d) for maxpool_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_maxpool_layer(layer_t *model, param_t p, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append maxpool layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  if ( tail->signal->dim != 3 ) {
    fprintf(stderr, "Error: failed to append maxpool layer\n");
    fprintf(stderr, "       Invalid input dimension (%ld)\n", tail->signal->dim);
    exit(EXIT_FAILURE);
  }
  int ich  = tail->signal->width[0];
  int irow = tail->signal->width[1];
  int icol = tail->signal->width[2];

  size_t *width   = (size_t *)xmalloc(sizeof(size_t)*3);
  width[0]        = ich;
  width[1]        = p.fsize;
  width[2]        = p.fsize;

  size_t *osigw   = (size_t *)xmalloc(sizeof(size_t)*3);
  int orow        = get_maxpool_osize(irow, p.fsize, p.pad, p.stride);
  int ocol        = get_maxpool_osize(irow, p.fsize, p.pad, p.stride);
  osigw[0]        = ich;
  osigw[1]        = orow;
  osigw[2]        = ocol;

  int osize       = ich * orow * ocol;
  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = 3;
  sig->width      = osigw;
  sig->signal     = (float *)xmalloc(sizeof(float)*osize*batch);

  int isize       = ich * irow * icol;
  float *delta    = (float *)xmalloc(sizeof(float)*isize*batch);

  layer->dim      = 3;
  layer->width    = width;
  layer->fn_ptr   = maxpool_layer;
  layer->stride   = p.stride;
  layer->pad      = p.pad;
  layer->weight   = NULL;
  layer->signal   = sig;
  layer->bias     = NULL;
  layer->delta    = delta;
  layer->wv       = NULL;
  layer->wm       = NULL;
  layer->bv       = NULL;
  layer->bm       = NULL;
  layer->learning_rate = 0;
  layer->beta1    = 0.0;
  layer->beta2    = 0.0;
  layer->strsize  = 0;
  layer->string   = NULL;

  tail->next = layer;
  layer->prev = tail;
  layer->next = NULL;

  return layer;
}
