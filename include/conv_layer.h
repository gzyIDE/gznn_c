#ifndef _CONV_LAYER_H_INCLUDED_
#define _CONV_LAYER_H_INCLUDED_

#include "model.h"

void conv_forward(layer_t *layer, size_t batch);
void conv_backward(layer_t *layer, size_t batch);
void conv_update(layer_t *layer, size_t batch);
void conv_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *add_conv_layer(layer_t *model, param_t p, size_t batch);

#endif //_CONV_LAYER_H_INCLUDED_
