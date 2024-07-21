#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "operation.h"

size_t get_conv_osize(size_t isize, size_t fsize, size_t pad, size_t stride) {
  return (isize + 2 * pad - fsize) / stride + 1;
}

void conv_forward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  int ich         = layer->width[1]; // must be equal to input->width[0];
  int fsize       = layer->width[2];
  //int frow        = layer->width[2];
  //int fcol        = layer->width[3];
  int pad         = layer->pad;
  int stride      = layer->stride;
  int irow        = input->width[1];
  int icol        = input->width[2];
  int och         = layer->signal->width[0];
  int orow        = layer->signal->width[1];
  int ocol        = layer->signal->width[2];
  int ielm        = fsize * fsize * ich;
  int oelm        = orow * ocol;
  float *data     = (float *)xmalloc(sizeof(float)*ielm * orow * ocol); //im2col result

  for (int b = 0; b < batch; b++) {
    float *in  = &input->signal[b*irow*icol*ich];
    float *out = &layer->signal->signal[b*orow*ocol*och];
    // weight: (och, fsize*fsize*ich)
    // data  : transpose(orow*ocol, fsize*fsize*ich)
    im2col(in, data, irow, icol, ich, fsize, pad, stride);
    matmul(layer->weight, data, out, och, ielm, oelm, ielm, 0, 1);

    for (int ch = 0; ch < och; ch++) {
      float *outch = &out[ch*orow*ocol];
      vbias(outch, outch, layer->bias[ch], orow*ocol);
    }
  }

  free(data);
}

void conv_backward(layer_t *layer, size_t batch) {
  signal_t *input = layer->prev->signal;
  int ich         = layer->width[1]; // must be equal to input->width[0];
  int fsize       = layer->width[2];
  //int frow        = layer->width[2];
  //int fcol        = layer->width[3];
  int pad         = layer->pad;
  int stride      = layer->stride;
  int irow        = input->width[1];
  int icol        = input->width[2];
  int och         = layer->signal->width[0];
  int orow        = layer->signal->width[1];
  int ocol        = layer->signal->width[2];
  int ielm        = irow * icol;
  int oelm        = fsize * fsize * och;
  float *flipw    = (float *)xmalloc(sizeof(float) * ich * och * fsize * fsize);
  float *data     = (float *)xmalloc(sizeof(float) * oelm * irow * icol); //im2col result

  // TODO: stride > 1のサポート

  // weight rearrange
  //   before: (och, ich, fsize, fsize)
  //   after : (ich, och, rotate180(fsize, fsize))
  int ksz = fsize* fsize;
  for (int i = 0; i < och; i++) {
    for (int j = 0; j < ich; j++) {
      for (int k = 0; k < ksz; k++) {
        int src = ksz*(i*ich+j) + (ksz-k-1);
        int dst = ksz*(j*och+i) + k;
        flipw[dst] = layer->weight[src];
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    float *delta_o = &layer->next->delta[b*orow*ocol*och];
    float *delta_i = &layer->delta[b*irow*icol*ich];

    // weight: (ich, fsize*fsize*och)
    // data : transpose(irow*icol, fsize*fsize*och) -> (fsize*fsize*och, irow*icol)
    im2col(delta_o, data, orow, ocol, och, fsize, fsize-1-pad, stride);
    matmul(flipw, data, delta_i, ich, oelm, ielm, oelm, 0, 1);
  }

  free(flipw);
  free(data);
}

void conv_update(layer_t *layer, int step, size_t batch) {
  signal_t *input   = layer->prev->signal;
  int och           = layer->width[0];
  int ich           = layer->width[1];
  int fsize         = layer->width[2];
  //int frow          = layer->width[2];
  //int fcol          = layer->width[3];
  int pad           = layer->pad;
  int stride        = layer->stride;
  int irow          = input->width[1];
  int icol          = input->width[2];
  int orow          = layer->signal->width[1];
  int ocol          = layer->signal->width[2];
  int ielm          = fsize * fsize * ich;
  int oelm          = orow * ocol;
  float *data       = (float *)xmalloc(sizeof(float)*ielm*orow*ocol); //im2col result

  float *grad_w     = (float *)xmalloc(sizeof(float)*fsize*fsize*ich*och);
  float *grad_w_sum = (float *)xcalloc(fsize*fsize*ich*och,sizeof(float));
  float *grad_b     = (float *)xcalloc(och, sizeof(float));

  // gradient calculation
  for (int b = 0; b < batch; b++) {
    float *in      = &input->signal[b*irow*icol*ich];
    float *delta_o = &layer->next->delta[b*orow*ocol*och];

    // delta_o : (och, orow*ocol)
    // data    : (orow*ocol, fsize*fsize*ich)
    im2col(in, data, irow, icol, ich, fsize, pad, stride);
    matmul(delta_o, data, grad_w, och, oelm, oelm, ielm, 0, 0);
    saxpy(1.0, grad_w, grad_w_sum, grad_w_sum, fsize*fsize*ich*och);

    for (int ch = 0; ch < och; ch++) {
      grad_b[ch] = vsum(&layer->next->delta[orow*ocol*ch], orow*ocol);
    }
  }

  // Update with Adam
  float b1   = layer->beta1;
  float b2   = layer->beta2;
  float rate = layer->learning_rate;
  adam(layer->wm, layer->wv, grad_w, layer->weight, b1, b2, rate, step, fsize*fsize*ich*och);
  adam(layer->bm, layer->bv, grad_b, layer->weight, b1, b2, rate, step, och);

  free(data);
  free(grad_w);
  free(grad_w_sum);
  free(grad_b);
}

void conv_print(layer_t *layer) {
  printf("conv_layer:\n");
  printf("  stride: %ld\n", layer->stride);
  printf("  pad   : %ld\n", layer->pad);
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

void conv_dump(layer_t *layer) {
}

void conv_layer(layer_t *layer, int step, size_t batch, int cmd) {
  //printf("conv layer\n");

  switch (cmd) {
    case FORWARD  : conv_forward(layer, batch); break;
    case BACKWARD : conv_backward(layer, batch); break;
    case UPDATE   : conv_update(layer, step, batch); break;
    case PRINT    : conv_print(layer); break;
    case DUMP     : conv_dump(layer); break;
    default : 
      fprintf(stderr, "Error: invalid command (%d) for fc_layer\n", cmd);
      exit(EXIT_FAILURE);
  }
}

layer_t *add_conv_layer(layer_t *model, param_t p, size_t batch) {
  layer_t *layer  = (layer_t *)xmalloc(sizeof(layer_t));

  if ( model == NULL ) {
    fprintf(stderr, "Error: failed to append convolution layer\n");
    fprintf(stderr, "       Invalid previous layer\n");
    exit(EXIT_FAILURE);
  }

  layer_t *tail   = find_last(model);

  if ( tail->signal->dim != 3 ) {
    fprintf(stderr, "Error: failed to append convolution layer\n");
    fprintf(stderr, "       Invalid input dimension (%ld)\n", tail->signal->dim);
    exit(EXIT_FAILURE);
  }
  int ich  = tail->signal->width[0];
  int irow = tail->signal->width[1];
  int icol = tail->signal->width[2];
  int ielm = ich * irow * icol;

  size_t *width   = (size_t *)xmalloc(sizeof(size_t)*4);
  width[0]        = p.och;
  width[1]        = ich;
  width[2]        = p.fsize;
  width[3]        = p.fsize;

  size_t *osigw   = (size_t *)xmalloc(sizeof(size_t)*3);
  int orow        = get_conv_osize(irow, p.fsize, p.pad, p.stride);
  int ocol        = get_conv_osize(icol, p.fsize, p.pad, p.stride);
  osigw[0]        = p.och;
  osigw[1]        = orow;
  osigw[2]        = ocol;

  int wsize       = ich * p.och * p.fsize * p.fsize;
  float *weight   = (float *)xmalloc(sizeof(float)*wsize);
  //float wscale    = 1.0 / sqrt(ielm);
  float wscale    = sqrt(2.0 / ielm);
  init_weight(4, width, weight, wscale);

  int bsize       = p.och;
  float *bias     = (float *)xmalloc(sizeof(float)*bsize);
  init_bias(bsize, bias);

  int osize       = p.och * orow * ocol;
  signal_t *sig   = (signal_t *)xmalloc(sizeof(signal_t));
  sig->dim        = 3;
  sig->width      = osigw;
  sig->signal     = (float *)xmalloc(sizeof(float)*osize*batch);

  int isize       = ich * irow * icol;
  float *delta    = (float *)xmalloc(sizeof(float)*isize*batch);

  layer->dim      = 4;
  layer->width    = width;
  layer->fn_ptr   = conv_layer;
  layer->stride   = p.stride;
  layer->pad      = p.pad;
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

