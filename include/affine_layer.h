#ifndef _AFFINE_LAYER_H_INCLUDED_
#define _AFFINE_LAYER_H_INCLUDED_

void affine_forward(layer_t *layer, size_t batch);
void affine_backward(layer_t *layer, size_t batch);
void affine_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *add_affine_layer(layer_t *model, param_t p, size_t batch);

#endif //_AFFINE_LAYER_H_INCLUDED_
