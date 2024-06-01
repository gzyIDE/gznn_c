#ifndef _FC_LAYER_H_INCLUDED_
#define _FC_LAYER_H_INCLUDED_

void fc_forward(layer_t *layer, float *input);
void fc_backward(layer_t *layer, float *input);
void fc_layer(layer_t *layer, float *input, int dir);

#endif //_FC_LAYER_H_INCLUDED_
