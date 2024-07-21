#ifndef _MAXPOOL_LAYER_H_INCLUDED_
#define _MAXPOOL_LAYER_H_INCLUDED_

void maxpool_forward(layer_t *layer, size_t batch);
void maxpool_backward(layer_t *layer, size_t batch);
void maxpool_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *add_maxpool_layer(layer_t *model, param_t p, size_t batch);

#endif //_MAXPOOL_LAYER_H_INCLUDED_
