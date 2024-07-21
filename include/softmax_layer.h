#ifndef _SOFTMAX_LAYER_H_INCLUDED_
#define _SOFTMAX_LAYER_H_INCLUDED_

#include "model.h"

void softmax_forward(layer_t *layer, size_t batch);
void softmax_backward(layer_t *layer, size_t batch);
void softmax_with_loss_delta(layer_t *layer, float *t, size_t batch);
void softmax_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *add_softmax_layer(layer_t *model, size_t batch);

#endif //_SOFTMAX_LAYER_H_INCLUDED_
