#ifndef _IDENTITY_LAYER_H_INCLUDED_
#define _IDENTITY_LAYER_H_INCLUDED_

void identity_forward(layer_t *layer, size_t batch);
void identity_backward(layer_t *layer, size_t batch);
void identity_layer(layer_t *layer, int step, size_t batch, int cmd);

#endif //_IDENTITY_LAYER_H_INCLUDED_
