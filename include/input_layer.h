#ifndef _INPUT_LAYER_H_INCLUDED_
#define _INPUT_LAYER_H_INCLUDED_

void input_layer(layer_t *layer, int step, size_t batch, int cmd);
layer_t *generate_input_layer(int ch, int row, int col, int batch);

#endif //_INPUT_LAYER_H_INCLUDED_
