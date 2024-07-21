#ifndef _RELU_LAYER_H_INCLUDED_
#define _RELU_LAYER_H_INCLUDED_

#include "model.h"

void relu_forward(layer_t *layer, size_t batch);
void relu_backward(layer_t *layer, size_t batch);
void relu_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *add_relu_layer(layer_t *model, size_t batch);

#endif //_RELU_LAYER_H_INCLUDED_
