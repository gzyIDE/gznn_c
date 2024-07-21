#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "layers.h"
#include "operation.h"
#include "cross_entropy.h"

void forward(layer_t *model, dataset_t *data, size_t batch) {
  int elm = data->width[0];
  for (int i = 1; i < data->dim; i++) elm *= data->width[i];

  // input data normalization
  model->signal->dim    = data->dim;
  model->signal->width  = data->width;
  model->signal->signal = normalize(data, 255.0, elm, batch);

  // forward
  layer_t  *current = model->next;
  do {
    current->fn_ptr(current, 0, batch, FORWARD);
  } while (current = current->next);
}

float backward(layer_t *model, dataset_t *data, size_t batch) {
  int elm = data->width[0];
  for (int i = 1; i < data->dim; i++) elm *= data->width[i];

  // input data normalization
  model->signal->signal = normalize(data, 255.0, elm, batch);

  // find output layer
  layer_t *current = find_last(model);

  // output delta calculation (Softmax-with loss layer)
  size_t tvec_w = current->width[0];
  float *tvec   = label2vec(data, tvec_w, batch);
  float *res    = current->signal->signal;
  float loss    = cross_entropy(tvec, res, tvec_w, batch);

  // back propagation
  softmax_with_loss_delta(current, tvec, batch);
  while (current = current->prev) {
    current->fn_ptr(current, 0, batch, BACKWARD);
  } 

  free(tvec);

  return loss;
}

void update(layer_t *model, dataset_t *data, int step, size_t batch) {
  size_t elm = data->width[0];
  for (int i = 1; i < data->dim; i++) elm *= data->width[i];

  // update
  layer_t  *current = model->next;
  do {
    current->fn_ptr(current, step, batch, UPDATE);
  } while (current = current->next);
}

float *normalize(dataset_t *data, float scale, size_t w, size_t batch) {
  float *normd = (float *)xmalloc(sizeof(float)*w*batch);
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < w; i++) {
      normd[b*w+i] = (float)data[b].data[i]/scale;
    }
  }

  return normd;
}

float *label2vec(dataset_t *data, size_t w, size_t batch) {
  float *tvec = (float *)xmalloc(sizeof(float) * w * batch);

  for ( int b = 0; b < batch; b++ ) {
    for ( int i = 0; i < w; i++ ) {
      tvec[b*w+i] = (i == data[b].label) ? 1.0 : 0.0;
    }
  }

  return tvec;
}

layer_t *find_first(layer_t *model) {
  layer_t *current = model;
  while(current->prev) current = current->prev;
  return current;
}

layer_t *find_last(layer_t *model) {
  layer_t *current =  model;
  while (current->next) current = current->next;
  return current;
}

float box_muller(void) {
  float x = (float)rand() / (float)RAND_MAX;
  float y = (float)rand() / (float)RAND_MAX;
  float z = sqrtf(-2.0 * logf(x)) * cosf(2 * M_PI * y);
  return z;
}

void init_weight(size_t dim, size_t *w, float *weight, float scale) {
  int elm = w[0];
  for (int i = 1; i < dim; i++) elm *= w[i];

  for (int i = 0; i < elm; i++) {
    weight[i] = box_muller() * scale;
  }
}

void init_bias(size_t w, float *bias) {
  memset(bias, 0, sizeof(float) * w);
}

void print_layer(layer_t *model) {
  layer_t *current = find_first(model);

  while (current) {
    current->fn_ptr(current, 0, 0, PRINT);
    current = current->next;
  }
}

void dump_model(layer_t *model) {
  layer_t *current = find_first(model);

  while(current) {
    current->fn_ptr(current, 0, 0, PRINT);
    current = current->next;
  }
}
